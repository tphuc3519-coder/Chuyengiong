"use client";

import { useId, useRef, useState } from "react";

import { formatBytes } from "@/lib/params";

/**
 * Drag-and-drop plus a file picker, because on iOS — the device this is tested
 * on first — dragging does not exist and the picker is the whole interaction.
 * The visible control is a `<label>` for a real `<input type="file">` rather
 * than a div with a click handler, so the keyboard and VoiceOver get it free.
 */
export function FileDrop({
  file,
  onFile,
  accept,
  maxBytes,
  label,
  hint,
  disabled,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
  accept: string;
  maxBytes: number;
  label: string;
  hint: string;
  disabled?: boolean;
}) {
  const inputId = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function accepted(candidate: File | undefined | null) {
    setError(null);
    if (!candidate) return;
    if (candidate.size > maxBytes) {
      setError(`File ${formatBytes(candidate.size)}, giới hạn ${formatBytes(maxBytes)}`);
      return;
    }
    if (candidate.size === 0) {
      setError("File rỗng");
      return;
    }
    onFile(candidate);
  }

  function clear() {
    onFile(null);
    setError(null);
    if (input.current) input.current.value = "";
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className={`drop${over ? " is-over" : ""}${file ? " has-file" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          if (!disabled) accepted(event.dataTransfer.files?.[0]);
        }}
      >
        <input
          id={inputId}
          ref={input}
          type="file"
          accept={accept}
          disabled={disabled}
          className="visually-hidden"
          onChange={(event) => accepted(event.target.files?.[0])}
        />
        {file ? (
          <span className="drop-body">
            <strong className="drop-filename">{file.name}</strong>
            <span className="drop-hint">{formatBytes(file.size)} · bấm để đổi file</span>
          </span>
        ) : (
          <span className="drop-body">
            <strong>{label}</strong>
            <span className="drop-hint">{hint}</span>
          </span>
        )}
      </label>
      {file && !disabled && (
        <button type="button" className="link-button" onClick={clear}>
          Bỏ file
        </button>
      )}
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
