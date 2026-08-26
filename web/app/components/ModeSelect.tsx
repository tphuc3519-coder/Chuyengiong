"use client";

import { MODES, type Mode } from "@/lib/params";

export function ModeSelect({
  value,
  onChange,
  disabled,
}: {
  value: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label="Kiểu nội dung">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          type="button"
          role="radio"
          aria-checked={value === mode.id}
          className={value === mode.id ? "segment is-active" : "segment"}
          disabled={disabled}
          onClick={() => onChange(mode.id)}
        >
          <span className="segment-label">{mode.label}</span>
          <span className="segment-hint">{mode.hint}</span>
        </button>
      ))}
    </div>
  );
}
