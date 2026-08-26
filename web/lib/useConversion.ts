"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  POLL_TIMEOUT_MS,
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
  jobsRemaining: number | null;
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
  jobsRemaining: null,
  error: null,
  blob: null,
};

const MAX_POLL_FAILURES = 5;

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
    return "Đã dùng hết lượt trong giờ này (5 bài/giờ). Thử lại sau nhé.";
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

        setState((current) => ({
          ...current,
          phase: "running",
          status: submitted.status,
          uploaded: 1,
          jobId: submitted.job_id,
          jobsRemaining: submitted.jobs_remaining ?? null,
        }));

        const record = await watch(submitted.job_id, controller.signal, (update) =>
          setState((current) => ({
            ...current,
            status: update.status,
            progress: update.progress,
          })),
        );

        const blob = await download(record.id, controller.signal);
        setState((current) => ({
          ...current,
          phase: "done",
          status: "done",
          progress: 100,
          blob,
        }));
      } catch (error) {
        if (aborted(error)) return; // the user left or started another run
        setState((current) => ({ ...current, phase: "error", error: message(error) }));
      } finally {
        if (abort.current === controller) abort.current = null;
      }
    },
    [cancel],
  );

  return { state, start, reset, cancel };
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
      throw new ApiError(0, "Quá 15 phút mà chưa xong — có thể job đã hỏng, thử lại nhé.");
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
