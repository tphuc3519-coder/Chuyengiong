"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  POLL_TIMEOUT_MS,
  apiBase,
  download,
  poll,
  pollDelay,
  submit,
  type JobRecord,
  type Status,
  type SubmitInput,
} from "./api";

/**
 * The run: upload → poll → fetch the result, as one state machine.
 *
 * Everything cancellable hangs off a single AbortController that the unmount
 * effect trips, so navigating away mid-run does not leave a poll loop and an
 * in-flight upload behind (plan §6).
 *
 * A poll that fails is not a job that failed. The network drops on mobile, and
 * a job that is three minutes into a GPU conversion should survive it, so
 * failures are counted and only give up after several in a row.
 */

export type Phase = "idle" | "uploading" | "running" | "done" | "error";

export type RunState = {
  phase: Phase;
  status: Status | "uploading";
  progress: number;
  uploaded: number;
  elapsedMs: number;
  jobId: string | null;
  /**
   * The result straight off Modal, for the browser to fetch by itself.
   *
   * The in-page download reads megabytes into a blob, which is the step that
   * dies on a phone. A plain link is the same file by a route that has already
   * been shown to work when the fetch does not — no blob, no CORS, no page
   * left holding the failure.
   */
  resultUrl: string | null;
  jobsRemaining: number | null;
  /** The shift the backend measured, once it has. Null while auto-detect runs. */
  semitoneShift: number | null;
  error: string | null;
  blob: Blob | null;
};

const IDLE: RunState = {
  phase: "idle",
  status: "queued",
  progress: 0,
  uploaded: 0,
  elapsedMs: 0,
  jobId: null,
  resultUrl: null,
  jobsRemaining: null,
  semitoneShift: null,
  error: null,
  blob: null,
};

const MAX_POLL_FAILURES = 5;
const MAX_DOWNLOAD_ATTEMPTS = 4;

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function aborted(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function message(error: unknown): string {
  if (error instanceof ApiError && error.status === 429) {
    // No number: the cap is whatever `JOBS_PER_HOUR` says on the deployment,
    // and quoting five here was already wrong once.
    return "Đã dùng hết lượt trong giờ này. Thử lại sau nhé.";
  }
  // A `fetch` that never reached the server rejects with a TypeError, and the
  // text it carries is the browser's own: Safari says "Load failed", Chrome
  // "Failed to fetch". Neither names what broke or what to do about it, and
  // showing it verbatim is how a dropped connection came to read as a bug.
  if (error instanceof TypeError) {
    return "Mất kết nối tới máy chủ. Kiểm tra mạng rồi thử lại.";
  }
  if (error instanceof Error && error.message) return error.message;
  return "Có lỗi không xác định.";
}

export function useConversion() {
  const [state, setState] = useState<RunState>(IDLE);
  const abort = useRef<AbortController | null>(null);
  const startedAt = useRef(0);

  const cancel = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
  }, []);

  useEffect(() => cancel, [cancel]);

  // One ticking clock for the whole run, so the elapsed time keeps moving even
  // while the poll interval has backed off to ten seconds.
  useEffect(() => {
    if (state.phase !== "uploading" && state.phase !== "running") return;
    const timer = setInterval(
      () => setState((current) => ({ ...current, elapsedMs: Date.now() - startedAt.current })),
      500,
    );
    return () => clearInterval(timer);
  }, [state.phase]);

  const reset = useCallback(() => {
    cancel();
    setState(IDLE);
  }, [cancel]);

  /**
   * Watch a job to the end. Shared with `resume`.
   *
   * The run is finished the moment the server says `done` — the file exists,
   * and the link to it is already in state. Fetching the bytes is a separate
   * thing that happens afterwards and is allowed to fail: `response.blob()`
   * reads megabytes into the page, which is exactly what an hours-old Safari
   * tab will not do, and that failure used to stand between the user and a
   * finished conversion. Now it costs them the inline player and nothing else.
   */
  const collect = useCallback(async (jobId: string, controller: AbortController) => {
    const record = await watch(jobId, controller.signal, (update) =>
      setState((current) => ({
        ...current,
        status: update.status,
        progress: update.progress,
        semitoneShift: update.semitone_shift ?? current.semitoneShift,
      })),
    );
    setState((current) => ({ ...current, phase: "done", status: "done", progress: 100 }));

    try {
      const blob = await fetchResult(record.id, controller.signal);
      setState((current) => ({ ...current, blob }));
    } catch (error) {
      if (aborted(error)) throw error;
      // Playing it here was the convenience; the link is the deliverable.
    }
  }, []);

  const start = useCallback(
    async (input: Omit<SubmitInput, "onProgress" | "signal">) => {
      cancel();
      const controller = new AbortController();
      abort.current = controller;
      startedAt.current = Date.now();
      setState({ ...IDLE, phase: "uploading", status: "uploading" });

      try {
        const submitted = await submit({
          ...input,
          signal: controller.signal,
          onProgress: (fraction) =>
            setState((current) =>
              current.phase === "uploading" ? { ...current, uploaded: fraction } : current,
            ),
        });

        // Already resolved — `submit` awaited it before sending a byte.
        const base = await apiBase();
        setState((current) => ({
          ...current,
          phase: "running",
          status: submitted.status,
          uploaded: 1,
          jobId: submitted.job_id,
          resultUrl: `${base}/download/${submitted.job_id}`,
          jobsRemaining: submitted.jobs_remaining ?? null,
        }));

        await collect(submitted.job_id, controller);
      } catch (error) {
        if (aborted(error)) return; // the user left or started another run
        setState((current) => ({ ...current, phase: "error", error: message(error) }));
      } finally {
        if (abort.current === controller) abort.current = null;
      }
    },
    [cancel, collect],
  );

  /**
   * Pick a job back up after the connection dropped.
   *
   * A lost connection is not a lost conversion: the result sits on the server
   * for six hours. On a phone it is also the likely ending for a job that runs
   * for minutes — iOS suspends a backgrounded tab and takes the poll loop's
   * timers and sockets with it, so the wake-up finds a dead fetch. Without
   * this the only route back to a finished file was reading a job id out of
   * the server's own logs, which is not something to ask of anyone.
   */
  const resume = useCallback(
    async (jobId: string) => {
      cancel();
      const controller = new AbortController();
      abort.current = controller;
      // A fresh clock: this watch starts now, and carrying the original run's
      // start forward showed 45:00 on a poll two seconds old.
      startedAt.current = Date.now();
      setState((current) => ({ ...current, phase: "running", error: null, elapsedMs: 0 }));

      try {
        await collect(jobId, controller);
      } catch (error) {
        if (aborted(error)) return;
        setState((current) => ({ ...current, phase: "error", error: message(error) }));
      } finally {
        if (abort.current === controller) abort.current = null;
      }
    },
    [cancel, collect],
  );

  return { state, start, resume, reset, cancel };
}

/**
 * Fetch the finished mp3, riding out a dropped connection.
 *
 * `watch` already survives five failed polls, on the reasoning that the network
 * drops on mobile and a job minutes into a GPU should not die of it. The same
 * reasoning applies harder here and had no retry at all: this is megabytes over
 * a phone connection rather than a few hundred bytes, it is the very last step,
 * and the result it was throwing away sits on the server for six hours. One
 * blip turned a finished job into Safari's "Load failed".
 */
async function fetchResult(jobId: string, signal: AbortSignal): Promise<Blob> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await download(jobId, signal);
    } catch (error) {
      if (aborted(error)) throw error;
      // A 4xx is an answer, not a dropped connection: the job is unknown or its
      // files have expired, and asking again will say the same thing.
      const fatal = error instanceof ApiError && error.status >= 400 && error.status < 500;
      if (fatal || attempt >= MAX_DOWNLOAD_ATTEMPTS) throw error;
      await sleep(attempt * 1000, signal);
    }
  }
}

async function watch(
  jobId: string,
  signal: AbortSignal,
  onUpdate: (record: JobRecord) => void,
): Promise<JobRecord> {
  const began = Date.now();
  let failures = 0;

  for (;;) {
    const elapsed = Date.now() - began;
    if (elapsed > POLL_TIMEOUT_MS) {
      throw new ApiError(0, "Quá 30 phút mà chưa xong — job đã bị dừng, thử lại nhé.");
    }
    await sleep(pollDelay(elapsed), signal);

    let record: JobRecord;
    try {
      record = await poll(jobId, signal);
      failures = 0;
    } catch (error) {
      if (aborted(error)) throw error;
      // A 4xx is an answer, not a dropped connection: the job is unknown or
      // its record expired, and four more polls will say the same thing.
      // Only 5xx and network errors are worth waiting out.
      const fatal = error instanceof ApiError && error.status >= 400 && error.status < 500;
      if (fatal || ++failures >= MAX_POLL_FAILURES) throw error;
      continue;
    }

    onUpdate(record);
    if (record.status === "done") return record;
    if (record.status === "failed") {
      throw new ApiError(0, record.error ?? "Job thất bại mà không kèm lý do.");
    }
  }
}
