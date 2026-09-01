"use client";

import { useId, useState } from "react";

import {
  DIFFUSION_STEPS_MAX,
  DIFFUSION_STEPS_MIN,
  MAX_SEMITONE_SHIFT,
  MAX_VOCAL_GAIN_DB,
  SPEAKING_RATE_MAX,
  SPEAKING_RATE_MIN,
  clamp,
  formatRate,
  formatSemitones,
  type Mode,
  type Params,
} from "@/lib/params";

/**
 * Collapsed by default (plan §6 step 4). Most runs should be Convert-and-wait;
 * the sliders are for the second attempt, once something sounded wrong.
 *
 * Pitch is the exception to "a slider with a default": the useful default is
 * measured off the vocal stem during the run (plan §7), so before the first
 * run there is no number to show. Auto is therefore a mode rather than a
 * starting value, and `detected` — what the last run actually applied — is
 * what the slider opens on when the user turns auto off.
 */
export function Advanced({
  mode,
  params,
  detected,
  onChange,
  disabled,
}: {
  mode: Mode;
  params: Params;
  detected: number | null;
  onChange: (params: Params) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const pitchLimit = MAX_SEMITONE_SHIFT[mode];
  const auto = params.semitoneShift === null;
  // What "no explicit shift" means differs by mode, because the backend only
  // measures one for speech (`AUTO_DETECT_MODES` in `pipeline.py`, which `tts`
  // reaches by converting as speech). A song's vocal is mixed back over an
  // instrumental nothing transposes, so leaving the key alone is the answer
  // there rather than a measurement.
  const keepsKey = mode === "song";

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
        {mode === "tts" && (
          <label className="slider">
            <span className="slider-label">
              Tốc độ đọc
              <output>{formatRate(params.speakingRate)}</output>
            </span>
            <input
              type="range"
              min={SPEAKING_RATE_MIN}
              max={SPEAKING_RATE_MAX}
              step={0.05}
              value={params.speakingRate}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...params, speakingRate: Number(event.target.value) })
              }
            />
            <span className="slider-hint">
              Áp dụng lúc đọc văn bản, trước khi đổi giọng — nên nhịp nói là của bản đọc này, không
              phải của giọng mẫu.
            </span>
          </label>
        )}

        <div className="slider">
          <span className="slider-label">
            Dịch cao độ
            <output>
              {auto
                ? keepsKey
                  ? "giữ tone gốc"
                  : detected === null
                    ? "tự động"
                    : `tự động · lần trước ${formatSemitones(detected)}`
                : `${formatSemitones(params.semitoneShift ?? 0)} nửa cung`}
            </output>
          </span>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={auto}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...params,
                  // Turning auto off starts from what the last run measured,
                  // which is the number the user is reacting to.
                  semitoneShift: event.target.checked
                    ? null
                    : clamp(detected ?? 0, -pitchLimit, pitchLimit),
                })
              }
            />
            <span>{keepsKey ? "Giữ nguyên tone gốc" : "Tự động dò từ giọng mẫu"}</span>
          </label>

          {!auto && (
            <input
              type="range"
              min={-pitchLimit}
              max={pitchLimit}
              step={1}
              value={params.semitoneShift ?? 0}
              disabled={disabled}
              aria-label="Dịch cao độ, nửa cung"
              onChange={(event) =>
                onChange({ ...params, semitoneShift: Number(event.target.value) })
              }
            />
          )}

          <span className="slider-hint">
            {auto
              ? keepsKey
                ? "Nhạc nền không được dịch theo, nên một lượng dịch lẻ sẽ đặt giọng vào tông khác bản phối. Để nguyên là an toàn."
                : "Đo F0 trung vị của giọng trong bài và của giọng mẫu rồi lấy chênh lệch. Chỉ tính trên đoạn có tiếng."
              : keepsKey
                ? "Nam→nữ thường +12, nữ→nam thường −12. Bội số của 12 là một quãng tám, giữ đúng key với nhạc nền; các giá trị khác sẽ lệch."
                : `Giọng nói giới hạn ±${pitchLimit}: dịch xa hơn nghe méo thanh điệu.`}
          </span>
        </div>

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
