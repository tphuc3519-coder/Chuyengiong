"use client";

import { useId } from "react";

import { FileDrop } from "./FileDrop";
import {
  AUDIO_ACCEPT,
  BEAT_PROMPT_CHARS,
  BEAT_PROMPT_EXAMPLES,
  BEAT_RANDOM_SEED,
  MAX_INPUT_BYTES,
  type Params,
} from "@/lib/params";

/**
 * Where the replacement backing track comes from, on the `beat` branch.
 *
 * Two sources and exactly one of them — the backend refuses a job carrying
 * both, so this is a radio group rather than two independent fields. Upload is
 * the default because it is the one that always works: it needs no GPU, no
 * gated weights, and the licence of what comes out is whatever licence the
 * user chose when they got the file.
 *
 * The note under the generator is not a disclaimer, it is the thing most
 * likely to disappoint somebody: a generated loop can be in the right key and
 * at the right tempo and still clash where the vocal moves through chord
 * changes it cannot know about. It works on rap and most electronic music and
 * gets worse the more melodic the singing is, and that is a property of
 * putting a loop under a melody rather than something to be tuned away.
 */
export function BeatSource({
  params,
  beat,
  onBeat,
  onChange,
  disabled,
}: {
  params: Params;
  beat: File | null;
  onBeat: (file: File | null) => void;
  onChange: (params: Params) => void;
  disabled?: boolean;
}) {
  const groupId = useId();
  const promptId = useId();
  const generating = params.beatSource === "generate";
  const left = BEAT_PROMPT_CHARS - params.beatPrompt.length;

  return (
    <div>
      <span className="slider-label" id={groupId}>
        Beat mới lấy từ đâu
      </span>
      <div className="segmented" role="radiogroup" aria-labelledby={groupId}>
        <button
          type="button"
          role="radio"
          aria-checked={!generating}
          className={generating ? "segment" : "segment is-active"}
          disabled={disabled}
          onClick={() => onChange({ ...params, beatSource: "upload" })}
        >
          <span className="segment-label">Tải beat lên</span>
          <span className="segment-hint">Beat bạn đã có sẵn quyền dùng</span>
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={generating}
          className={generating ? "segment is-active" : "segment"}
          disabled={disabled}
          onClick={() => onChange({ ...params, beatSource: "generate" })}
        >
          <span className="segment-label">Tự sinh beat</span>
          <span className="segment-hint">Mô tả kiểu nhạc, máy làm beat mới</span>
        </button>
      </div>

      {generating ? (
        <>
          <label className="slider" htmlFor={promptId}>
            <span className="slider-label">
              Mô tả beat
              <output>còn {left} ký tự</output>
            </span>
            <textarea
              id={promptId}
              className="composer-text"
              rows={3}
              maxLength={BEAT_PROMPT_CHARS}
              value={params.beatPrompt}
              disabled={disabled}
              placeholder={BEAT_PROMPT_EXAMPLES[0]}
              onChange={(event) => onChange({ ...params, beatPrompt: event.target.value })}
            />
            <span className="slider-hint">
              Không cần ghi đúng BPM — beat sinh ra sẽ được đo lại rồi kéo về đúng tốc độ và tông
              của bài, nên phần mô tả chỉ cần đúng <em>chất</em> nhạc. Ví dụ:{" "}
              {BEAT_PROMPT_EXAMPLES.slice(1).join(" · ")}
            </span>
          </label>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={params.beatSeed !== BEAT_RANDOM_SEED}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...params, beatSeed: event.target.checked ? 7 : BEAT_RANDOM_SEED })
              }
            />
            <span>Cố định beat (chạy lại ra đúng beat cũ)</span>
          </label>

          <p className="field-note">
            Beat sinh ra không biết vòng hợp âm của bài, nên nó hợp với rap, hip-hop và nhạc điện tử
            — chỗ nào giọng hát đi giai điệu nhiều thì dễ chỏi ở những ô nhịp đổi hợp âm.
          </p>
        </>
      ) : (
        <FileDrop
          file={beat}
          onFile={onBeat}
          accept={AUDIO_ACCEPT}
          maxBytes={MAX_INPUT_BYTES}
          label="Beat thay thế"
          hint="Kéo thả hoặc bấm để chọn · beat sẽ được kéo về đúng tốc độ và tông của bài"
          disabled={disabled}
        />
      )}
    </div>
  );
}
