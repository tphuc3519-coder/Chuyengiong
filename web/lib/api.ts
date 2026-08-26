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

import type { Mode, Params } from "./params";

export type Status = "queued" | "separating" | "converting" | "mixing" | "done" | "failed";

export type JobRecord = {
  id: string;
  status: Status;
  progress: number;
  mode: Mode;
  error: string | null;
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

let apiBasePromise: Promise<string> | null = null;

/** The Modal base URL, fetched once per page load and then remembered. */
export function apiBase(): Promise<string> {
  apiBasePromise ??= fetch("/api/config", { cache: "no-store" })
    .then(async (response) => {
      const body = await response.json().catch(() => ({}));
      if (!response.ok || !body?.apiBase) {
        throw new ApiError(response.status, body?.error ?? "converter is not configured");
      }
      return body.apiBase as string;
    })
    .catch((error) => {
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
  source: File;
  reference: File | Blob;
  referenceName: string;
  consent: boolean;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
};

export function submit(input: SubmitInput): Promise<SubmitResult> {
  const form = new FormData();
  form.set("input", input.source, input.source.name);
  form.set("reference", input.reference, input.referenceName);
  form.set("mode", input.mode);
  form.set("semitone_shift", String(input.params.semitoneShift));
  form.set("diffusion_steps", String(input.params.diffusionSteps));
  form.set("vocal_gain_db", String(input.params.vocalGainDb));
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
              reject(new ApiError(request.status, "the converter sent an unreadable reply"));
            }
            return;
          }
          reject(
            new ApiError(
              request.status,
              detail(request.responseText, `upload failed (${request.status})`),
            ),
          );
        };
        request.onerror = () =>
          reject(new ApiError(0, "could not reach the converter — check your connection"));
        request.ontimeout = () => reject(new ApiError(0, "the upload timed out"));
        request.onabort = () => reject(new DOMException("aborted", "AbortError"));

        input.signal?.addEventListener("abort", () => request.abort(), { once: true });
        request.send(form);
      }),
  );
}

export async function poll(jobId: string, signal?: AbortSignal): Promise<JobRecord> {
  const response = await fetch(`/api/status/${jobId}`, { cache: "no-store", signal });
  const body = await response.text();
  if (!response.ok) {
    throw new ApiError(response.status, detail(body, `status check failed (${response.status})`));
  }
  return JSON.parse(body) as JobRecord;
}

export async function download(jobId: string, signal?: AbortSignal): Promise<Blob> {
  const base = await apiBase();
  const response = await fetch(`${base}/download/${jobId}`, { signal });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      detail(await response.text(), "could not fetch the result"),
    );
  }
  return response.blob();
}

// --- polling schedule -----------------------------------------------------

export const POLL_INTERVAL_MS = 2000;
export const POLL_BACKOFF_AFTER_MS = 60_000;
export const POLL_MAX_INTERVAL_MS = 10_000;
export const POLL_TIMEOUT_MS = 15 * 60_000;

/**
 * Plan §6: every 2 seconds, backing off after a minute, giving up at fifteen.
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
  converting: "Đang đổi giọng",
  mixing: "Đang ghép lại",
  done: "Xong",
  failed: "Thất bại",
};
