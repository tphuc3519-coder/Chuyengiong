/**
 * Everything the browser does over the network.
 *
 * Three calls, on two different origins:
 *
 * * `submit` goes **straight to Modal**. A 3 minute mp3 is 4–7 MB and the
 *   serverless body limit is 4.5 MB, so routing the upload through this app
 *   would fail on exactly the files it is built for (plan §6). It uses
 *   XMLHttpRequest rather than fetch for one reason: upload progress. `fetch`
 *   still cannot report it, and a 7 MB upload on mobile data with no feedback
 *   looks broken.
 * * `poll` goes through `/api/status`, same origin as the page.
 * * `download` goes straight to Modal again — same size argument as the upload.
 */

import { convertsVoice, type Mode, type Params } from "./params";

export type Status =
  | "queued"
  | "separating"
  | "synthesizing"
  | "generating"
  | "converting"
  | "mixing"
  | "done"
  | "failed";

export type JobRecord = {
  id: string;
  status: Status;
  progress: number;
  mode: Mode;
  error: string | null;
  /**
   * What the conversion actually used. Null until separation is done and the
   * pitch measurement has run — with auto-detect this is the only place the
   * chosen value surfaces (plan §7).
   */
  semitone_shift: number | null;
};

export type SubmitResult = { job_id: string; status: Status; mode: Mode; jobs_remaining?: number };

/** An error carrying the HTTP status, so the UI can tell 429 from 500. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * What a `fetch` that never reached the server means, per request.
 *
 * `fetch` rejects with a bare `TypeError` for every network-level failure, and
 * the text it carries is the browser's own — "Load failed" on Safari. Reported
 * as one message it says only that something did not connect, which is not
 * enough to act on and was not enough to debug from either: three different
 * requests, three different situations, one indistinguishable error.
 *
 * The download line is the one that matters most. The result is on the server
 * for six hours at that point, so the run is not lost and the user should not
 * be told to start it again.
 */
const OFFLINE = {
  config: "Không lấy được địa chỉ máy chủ xử lý. Tải lại trang rồi thử lại.",
  status: "Mất liên lạc khi đang xử lý. Job có thể vẫn đang chạy — bấm lấy lại kết quả bên dưới.",
  download:
    "Đã xử lý xong nhưng chưa tải được file về. Kết quả còn trên máy chủ 6 giờ, thử lại nhé.",
} as const;

/**
 * `work`, with a bare network `TypeError` turned into a labelled failure.
 *
 * Takes a function, not a promise, so the whole exchange is inside it. Reading
 * the body is where a large download actually dies — `fetch` resolves as soon
 * as the headers land, and the megabytes arrive afterwards — so wrapping only
 * the call left the very failure this exists for reporting as the browser's
 * own "Load failed".
 */
async function offlineAs<T>(step: keyof typeof OFFLINE, work: () => Promise<T>): Promise<T> {
  try {
    return await work();
  } catch (error) {
    // Status 0 keeps the retry loops treating it as worth repeating: they stop
    // on a 4xx, which is an answer, and this is the absence of one.
    if (error instanceof TypeError) throw new ApiError(0, OFFLINE[step]);
    throw error;
  }
}

let apiBasePromise: Promise<string> | null = null;

// The smallest request the app makes, and the one every other request waits on:
// nothing can be uploaded or downloaded until it says where Modal is. `watch`
// rides out five failed polls and `fetchResult` four failed downloads, both
// because the network drops on mobile — and this had none, so a single blip
// before either of them got started ended the run.
const CONFIG_ATTEMPTS = 3;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchApiBase(): Promise<string> {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch("/api/config", { cache: "no-store" });
      const body = await response.json().catch(() => ({}));
      if (!response.ok || !body?.apiBase) {
        throw new ApiError(response.status, body?.error ?? "Máy chủ xử lý chưa được cấu hình.");
      }
      return body.apiBase as string;
    } catch (error) {
      // A reply is an answer, however unwelcome: MODAL_API_URL is unset on the
      // deployment and asking three times will not set it. Only a request that
      // never arrived is worth repeating.
      if (error instanceof ApiError || attempt >= CONFIG_ATTEMPTS) throw error;
      await wait(attempt * 400);
    }
  }
}

/** The Modal base URL, fetched once per page load and then remembered. */
export function apiBase(): Promise<string> {
  apiBasePromise ??= offlineAs("config", () => fetchApiBase()).catch((error) => {
    apiBasePromise = null; // let the next attempt retry rather than cache the failure
    throw error;
  });
  return apiBasePromise;
}

function detail(body: string, fallback: string): string {
  try {
    const parsed = JSON.parse(body);
    return typeof parsed?.detail === "string" ? parsed.detail : (parsed?.error ?? fallback);
  } catch {
    return fallback;
  }
}

export type SubmitInput = {
  mode: Mode;
  params: Params;
  /** The file to convert, or null on `tts` — there `text` is the input. */
  source: File | null;
  text?: string;
  /** `beat` mode only, and only when the user brought their own. */
  beat?: File | null;
  /** Omitted on `rebeat`, which converts nothing and refuses one. */
  reference: File | Blob | null;
  referenceName: string;
  consent: boolean;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
};

export function submit(input: SubmitInput): Promise<SubmitResult> {
  const form = new FormData();
  // One of the two, never both: `mode` is what decides which the backend reads,
  // and sending the other alongside it would only make a mismatch possible.
  if (input.mode === "tts") {
    form.set("text", input.text ?? "");
    form.set("language", input.params.language);
    form.set("speaking_rate", String(input.params.speakingRate));
    // How it is read. Both are clamped server-side and both have a default, so
    // a client that never learned about them still gets the natural reading.
    form.set("emotion", input.params.emotion);
    form.set("expressiveness", String(input.params.expressiveness));
  } else if (input.source) {
    form.set("input", input.source, input.source.name);
  }
  // Named rather than inferred: `remake` sends neither a file nor a prompt, so
  // there would be nothing for the backend to infer it from.
  if (input.mode === "beat" || input.mode === "rebeat") {
    form.set("beat_source", input.params.beatSource);
    if (input.params.beatSource === "generate") {
      form.set("beat_prompt", input.params.beatPrompt);
      form.set("beat_seed", String(input.params.beatSeed));
    } else if (input.params.beatSource === "remake") {
      form.set("arrange_style", input.params.arrangeStyle);
      form.set("beat_seed", String(input.params.beatSeed));
    } else if (input.beat) {
      form.set("beat", input.beat, input.beat.name);
    }
  }
  if (input.reference) form.set("reference", input.reference, input.referenceName);
  form.set("mode", input.mode);
  // Nothing about the conversion on the one mode that has none: the backend
  // does not read these for `rebeat`, and sending them would only suggest it
  // might.
  if (convertsVoice(input.mode)) {
    // Omitted, not sent as 0: an absent field is how the backend is told to
    // measure the pitch itself, and 0 is a real setting that suppresses it.
    if (input.params.semitoneShift !== null) {
      form.set("semitone_shift", String(input.params.semitoneShift));
    }
    form.set("diffusion_steps", String(input.params.diffusionSteps));
    // Always sent, never omitted: 0 is a real setting — no guidance at all —
    // so "leave it out and let the server decide" is a different request.
    form.set("cfg_rate", String(input.params.cfgRate));
  }
  form.set("vocal_gain_db", String(input.params.vocalGainDb));
  form.set("clarity", String(input.params.clarity));
  form.set("consent", String(input.consent));

  return apiBase().then(
    (base) =>
      new Promise<SubmitResult>((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("POST", `${base}/submit`);
        request.responseType = "text";

        request.upload.onprogress = (event) => {
          if (event.lengthComputable) input.onProgress?.(event.loaded / event.total);
        };
        request.onload = () => {
          if (request.status >= 200 && request.status < 300) {
            try {
              resolve(JSON.parse(request.responseText) as SubmitResult);
            } catch {
              reject(new ApiError(request.status, "Máy chủ trả về dữ liệu không đọc được."));
            }
            return;
          }
          reject(
            new ApiError(
              request.status,
              detail(request.responseText, `Tải file lên thất bại (${request.status}).`),
            ),
          );
        };
        request.onerror = () =>
          reject(new ApiError(0, "Không gửi được file lên máy chủ. Kiểm tra mạng rồi thử lại."));
        request.ontimeout = () =>
          reject(new ApiError(0, "Tải file lên quá lâu, đã dừng. Thử lại nhé."));
        request.onabort = () => reject(new DOMException("aborted", "AbortError"));

        if (input.signal?.aborted) {
          // Already cancelled before the send: the listener below would never
          // fire, and the upload would run to completion unwatched.
          reject(new DOMException("aborted", "AbortError"));
          return;
        }
        input.signal?.addEventListener("abort", () => request.abort(), { once: true });
        request.send(form);
      }),
  );
}

export async function poll(jobId: string, signal?: AbortSignal): Promise<JobRecord> {
  return offlineAs("status", async () => {
    const response = await fetch(`/api/status/${jobId}`, { cache: "no-store", signal });
    const body = await response.text();
    if (!response.ok) {
      throw new ApiError(
        response.status,
        detail(body, `Máy chủ trả lời lỗi khi hỏi trạng thái (${response.status}).`),
      );
    }
    return JSON.parse(body) as JobRecord;
  });
}

export async function download(jobId: string, signal?: AbortSignal): Promise<Blob> {
  const base = await apiBase();
  return offlineAs("download", async () => {
    const response = await fetch(`${base}/download/${jobId}`, { signal });
    if (!response.ok) {
      throw new ApiError(
        response.status,
        detail(await response.text(), "Không tải được kết quả từ máy chủ."),
      );
    }
    // The megabytes arrive here, not above: `fetch` resolved on the headers.
    return response.blob();
  });
}

// --- polling schedule -----------------------------------------------------

export const POLL_INTERVAL_MS = 2000;
export const POLL_BACKOFF_AFTER_MS = 60_000;
export const POLL_MAX_INTERVAL_MS = 10_000;
/**
 * Mirrored from `PIPELINE_TIMEOUT` in `modal_app/pipeline.py`.
 *
 * Deliberately not the fifteen minutes plan §6 asks for: the two numbers drifted
 * apart, and a client that gives up before the server does is the worse half of
 * the pair. It tells the user a running job failed and to try again, and trying
 * again starts a *second* GPU job beside the first — twice the bill and one more
 * slot off the hourly cap, for work that was going to arrive anyway. Fifteen
 * minutes is not even a generous ceiling: the input cap is fifteen minutes of
 * audio (`SOURCE_MAX_SEC`), which is minutes of GPU on its own before a cold
 * container has fetched its weights.
 *
 * So it waits exactly as long as the backend can run. Past this the pipeline
 * really has been killed, and "try again" is honest advice.
 */
export const POLL_TIMEOUT_MS = 30 * 60_000;

/**
 * Plan §6: every 2 seconds, backing off after a minute.
 *
 * The backoff is capped at 10s rather than left to double: separation and
 * conversion are minutes long, and a bar that can sit still for half a minute
 * reads as a hang.
 */
export function pollDelay(elapsedMs: number): number {
  if (elapsedMs < POLL_BACKOFF_AFTER_MS) return POLL_INTERVAL_MS;
  const doublings = Math.floor(elapsedMs / POLL_BACKOFF_AFTER_MS);
  return Math.min(POLL_INTERVAL_MS * 2 ** doublings, POLL_MAX_INTERVAL_MS);
}

export const STATUS_LABEL: Record<Status, string> = {
  queued: "Đang xếp hàng",
  separating: "Đang tách nhạc nền",
  synthesizing: "Đang đọc văn bản",
  generating: "Đang sinh beat",
  converting: "Đang đổi giọng",
  mixing: "Đang ghép lại",
  done: "Xong",
  failed: "Thất bại",
};
