"use client";

import { STATUS_LABEL, type Status } from "@/lib/api";
import { formatSeconds, formatSemitones } from "@/lib/params";

/**
 * One bar for the whole run, including the upload.
 *
 * The upload is its own phase because it is the part the user's connection
 * controls: on mobile data a 7 MB file can take longer than the conversion, and
 * a bar that only starts once the server has the file looks stuck for a minute.
 */
export function Progress({
  status,
  progress,
  uploaded,
  elapsedMs,
  semitoneShift,
}: {
  status: Status | "uploading";
  progress: number;
  uploaded: number;
  elapsedMs: number;
  semitoneShift: number | null;
}) {
  const uploading = status === "uploading";
  const percent = uploading ? Math.round(uploaded * 100) : progress;
  const label = uploading ? "Đang tải lên" : STATUS_LABEL[status];

  return (
    <div className="progress">
      <div className="progress-head">
        <span aria-live="polite">{label}…</span>
        <span className="progress-time">{formatSeconds(elapsedMs / 1000)}</span>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div
          className={`progress-fill${uploading ? " is-upload" : ""}`}
          style={{ width: `${Math.max(2, percent)}%` }}
        />
      </div>
      <p className="progress-hint">
        {uploading
          ? "Giữ tab này mở cho tới khi tải xong."
          : "Bài 3 phút thường mất 2–4 phút. Lần đầu chậm hơn vì máy chủ khởi động."}
      </p>
      {/* Appears mid-run: the shift is measured after separation, so this is
          the first moment there is anything to say about it. */}
      {semitoneShift !== null && (
        <p className="progress-hint">
          Cao độ: {formatSemitones(semitoneShift)} nửa cung
          {semitoneShift === 0 ? " (giữ nguyên)" : ""}
        </p>
      )}
    </div>
  );
}
