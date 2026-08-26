"use client";

import { useId, useState } from "react";

import {
  DIFFUSION_STEPS_MAX,
  DIFFUSION_STEPS_MIN,
  MAX_SEMITONE_SHIFT,
  MAX_VOCAL_GAIN_DB,
  type Mode,
  type Params,
} from "@/lib/params";

/**
 * Collapsed by default (plan §6 step 4). Most runs should be Convert-and-wait;
 * the sliders are for the second attempt, once something sounded wrong.
 */
export function Advanced({
  mode,
  params,
  onChange,
  disabled,
}: {
  mode: Mode;
  params: Params;
  onChange: (params: Params) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const pitchLimit = MAX_SEMITONE_SHIFT[mode];

  return (
    <div className="advanced">
      <button
        type="button"
        className="disclosure"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((was) => !was)}
      >
        <span className={open ? "chevron is-open" : "chevron"} aria-hidden="true" />
        Tinh chỉnh
      </button>

      <div id={panelId} hidden={!open} className="advanced-panel">
        <label className="slider">
          <span className="slider-label">
            Dịch cao độ
            <output>
              {params.semitoneShift > 0 ? "+" : ""}
              {params.semitoneShift} nửa cung
            </output>
          </span>
          <input
            type="range"
            min={-pitchLimit}
            max={pitchLimit}
            step={1}
            value={params.semitoneShift}
            disabled={disabled}
            onChange={(event) => onChange({ ...params, semitoneShift: Number(event.target.value) })}
          />
          <span className="slider-hint">
            {mode === "speech"
              ? `Giọng nói giới hạn ±${pitchLimit}: dịch xa hơn nghe méo thanh điệu.`
              : "Nam→nữ thường +12, nữ→nam thường −12. Để 0 nếu không chắc."}
          </span>
        </label>

        <label className="slider">
          <span className="slider-label">
            Chất lượng
            <output>{params.diffusionSteps} bước</output>
          </span>
          <input
            type="range"
            min={DIFFUSION_STEPS_MIN}
            max={DIFFUSION_STEPS_MAX}
            step={5}
            value={params.diffusionSteps}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...params, diffusionSteps: Number(event.target.value) })
            }
          />
          <span className="slider-hint">
            Cao hơn thì mượt hơn nhưng chậm hơn tuyến tính. Trên {DIFFUSION_STEPS_MAX} gần như không
            cải thiện thêm.
          </span>
        </label>

        {mode === "song" && (
          <label className="slider">
            <span className="slider-label">
              Âm lượng giọng
              <output>
                {params.vocalGainDb > 0 ? "+" : ""}
                {params.vocalGainDb} dB
              </output>
            </span>
            <input
              type="range"
              min={-MAX_VOCAL_GAIN_DB}
              max={MAX_VOCAL_GAIN_DB}
              step={1}
              value={params.vocalGainDb}
              disabled={disabled}
              onChange={(event) => onChange({ ...params, vocalGainDb: Number(event.target.value) })}
            />
            <span className="slider-hint">So với nhạc nền, trước khi chuẩn hoá độ ồn.</span>
          </label>
        )}
      </div>
    </div>
  );
}
