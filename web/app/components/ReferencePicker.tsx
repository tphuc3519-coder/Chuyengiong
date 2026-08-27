"use client";

import { useEffect, useState } from "react";

import { FileDrop } from "./FileDrop";
import { Recorder, recorderSupported } from "./Recorder";
import {
  AUDIO_ACCEPT,
  MAX_REFERENCE_BYTES,
  REFERENCE_MIN_SEC,
  REFERENCE_MAX_SEC,
} from "@/lib/params";
import { fetchPreset, loadPresets, type Preset } from "@/lib/presets";

type Source = "upload" | "record" | "preset";

export function ReferencePicker({
  file,
  onFile,
  disabled,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
  disabled?: boolean;
}) {
  const [source, setSource] = useState<Source>("upload");
  const [canRecord, setCanRecord] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Feature detection runs after mount: the server render has no `window`, and
  // guessing from the user agent gets iOS wrong in both directions.
  useEffect(() => setCanRecord(recorderSupported()), []);

  useEffect(() => {
    const abort = new AbortController();
    void loadPresets(abort.signal).then(setPresets);
    return () => abort.abort();
  }, []);

  async function pickPreset(preset: Preset) {
    setError(null);
    try {
      onFile(await fetchPreset(preset));
      setChosen(preset.id);
    } catch {
      setError("Không tải được giọng mẫu này.");
    }
  }

  const tabs: { id: Source; label: string; available: boolean }[] = [
    { id: "upload", label: "Tải lên", available: true },
    { id: "record", label: "Ghi âm", available: canRecord },
    { id: "preset", label: "Giọng có sẵn", available: presets.length > 0 },
  ];
  const visible = tabs.filter((tab) => tab.available);

  return (
    <div className="reference">
      {visible.length > 1 && (
        <div className="tabs" role="tablist" aria-label="Nguồn giọng mẫu">
          {visible.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={source === tab.id}
              className={source === tab.id ? "tab is-active" : "tab"}
              disabled={disabled}
              onClick={() => setSource(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}

      {source === "upload" && (
        <FileDrop
          file={file}
          onFile={(picked) => {
            setChosen(null);
            onFile(picked);
          }}
          accept={AUDIO_ACCEPT}
          maxBytes={MAX_REFERENCE_BYTES}
          label="Giọng mẫu"
          hint={`${REFERENCE_MIN_SEC}–${REFERENCE_MAX_SEC}s, một người nói, không nhạc nền`}
          disabled={disabled}
        />
      )}

      {source === "record" && canRecord && (
        <>
          <Recorder
            disabled={disabled}
            onRecorded={(recorded) => {
              setChosen(null);
              onFile(recorded);
            }}
          />
          {file && <p className="field-note">Đã có bản ghi — ghi lại sẽ thay bản cũ.</p>}
        </>
      )}

      {source === "preset" && (
        <ul className="presets">
          {presets.map((preset) => (
            <li key={preset.id}>
              <button
                type="button"
                className={chosen === preset.id ? "preset is-active" : "preset"}
                disabled={disabled}
                onClick={() => void pickPreset(preset)}
              >
                <strong>{preset.name}</strong>
                {preset.note && <span className="preset-note">{preset.note}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}

      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
