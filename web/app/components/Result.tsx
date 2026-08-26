"use client";

import { useEffect, useState } from "react";

export function Result({
  blob,
  jobId,
  onReset,
}: {
  blob: Blob;
  jobId: string;
  onReset: () => void;
}) {
  const [url, setUrl] = useState<string | null>(null);

  // One object URL per blob, revoked on the way out: without this a few runs in
  // a row hold every result in memory until the tab is closed.
  useEffect(() => {
    const objectUrl = URL.createObjectURL(blob);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [blob]);

  if (!url) return null;

  return (
    <div className="result">
      <audio className="player" controls preload="metadata" src={url} />
      <div className="result-actions">
        <a className="button" href={url} download={`voice-convert-${jobId.slice(0, 8)}.mp3`}>
          Tải về
        </a>
        <button type="button" className="button ghost" onClick={onReset}>
          Làm bài khác
        </button>
      </div>
      <p className="field-note">
        File có gắn metadata đánh dấu là nội dung tạo bởi AI, và bị xoá khỏi máy chủ sau 6 giờ.
      </p>
    </div>
  );
}
