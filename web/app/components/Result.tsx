"use client";

import { useEffect, useState } from "react";

import { formatSemitones } from "@/lib/params";

/**
 * The finished run.
 *
 * `resultUrl` is the file on the server and is always there; `blob` is a copy
 * read into the page, and is not. Reading it is what fails on a phone — an
 * hours-old Safari tab will not take megabytes into memory — so the download
 * button points at the server and the player is what goes missing when that
 * read does not come back. The file was never the part at risk.
 */
export function Result({
  blob,
  resultUrl,
  jobId,
  semitoneShift,
  onReset,
}: {
  blob: Blob | null;
  resultUrl: string;
  jobId: string;
  semitoneShift: number | null;
  onReset: () => void;
}) {
  const [url, setUrl] = useState<string | null>(null);

  // One object URL per blob, revoked on the way out: without this a few runs in
  // a row hold every result in memory until the tab is closed.
  useEffect(() => {
    if (!blob) return;
    const objectUrl = URL.createObjectURL(blob);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [blob]);

  return (
    <div className="result">
      {url ? (
        <audio className="player" controls preload="metadata" src={url} />
      ) : (
        <p className="field-note">
          Bấm &ldquo;Tải về&rdquo; để lấy file — nghe thử ngay tại đây chưa dùng được trên máy này.
        </p>
      )}
      <div className="result-actions">
        <a className="button" href={resultUrl} download={`voice-convert-${jobId.slice(0, 8)}.mp3`}>
          Tải về
        </a>
        <button type="button" className="button ghost" onClick={onReset}>
          Làm bài khác
        </button>
      </div>
      {semitoneShift !== null && (
        <p className="field-note">
          Đã dịch cao độ {formatSemitones(semitoneShift)} nửa cung. Nghe chưa vừa thì mở &ldquo;Tinh
          chỉnh&rdquo; ở lần sau và tự đặt lại.
        </p>
      )}
      <p className="field-note">
        File có gắn metadata đánh dấu là nội dung tạo bởi AI, và bị xoá khỏi máy chủ sau 6 giờ.
      </p>
    </div>
  );
}
