"use client";

import { useId } from "react";

import { FileDrop } from "./FileDrop";
import {
  ARRANGE_STYLES,
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
 * Three sources and exactly one of them, so this is a radio group rather than
 * three independent fields. Upload is the default because it is the one that
 * always works: no GPU, no gated weights, and the licence of what comes out is
 * whatever licence the user already had.
 *
 * The notes under the other two are not disclaimers — each is the thing most
 * likely to disappoint the person who picked it, and saying it once here is
 * cheaper than every user finding out the same way.
 *
 * For **generate**: a loop can be in the right key and at the right tempo and
 * still clash where the vocal moves through chord changes it cannot know
 * about. That is a property of putting a loop under a melody, not a setting.
 *
 * For **remake**: it removes the *recording* and not the *song*. The chords are
 * still the original's and the voice on top is still singing the original's
 * melody, which is what a cover is. The gain is real and narrow: covers are
 * licensable cheaply, masters usually are not licensable at all.
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
  const remaking = params.beatSource === "remake";
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
        <button
          type="button"
          role="radio"
          aria-checked={remaking}
          className={remaking ? "segment is-active" : "segment"}
          disabled={disabled}
          onClick={() => onChange({ ...params, beatSource: "remake" })}
        >
          <span className="segment-label">Phối lại bài này</span>
          <span className="segment-hint">Giữ nguyên tông và vòng hợp âm, thay hết tiếng</span>
        </button>
      </div>

      {remaking ? (
        <>
          <div className="slider">
            <span className="slider-label" id={`${groupId}-style`}>
              Kiểu phối
            </span>
            <div className="segmented" role="radiogroup" aria-labelledby={`${groupId}-style`}>
              {ARRANGE_STYLES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="radio"
                  aria-checked={params.arrangeStyle === item.id}
                  className={params.arrangeStyle === item.id ? "segment is-active" : "segment"}
                  disabled={disabled}
                  onClick={() => onChange({ ...params, arrangeStyle: item.id })}
                >
                  <span className="segment-label">{item.label}</span>
                  <span className="segment-hint">{item.hint}</span>
                </button>
              ))}
            </div>
            <span className="slider-hint">
              Máy đo tốc độ, tông và vòng hợp âm của bài gốc rồi chơi lại đúng vòng đó bằng tiếng tự
              tổng hợp — không một mẫu nào của bản gốc còn lại. Chỗ nào đọc hợp âm không chắc thì nó
              chỉ chơi trống và bass, vì trống thì không thể sai tông.
            </span>
          </div>

          <p className="field-note">
            Cách này gỡ được quyền <strong>bản ghi</strong> — không còn dùng bản thu của người ta.
            Nó <strong>không</strong> gỡ được quyền <strong>tác phẩm</strong>: hợp âm vẫn là của bài
            gốc và giọng vẫn hát đúng giai điệu đó, nên kết quả là một bản cover. Cái được thật sự
            là cover thì xin license được và rẻ, còn license bản ghi thì thường không xin nổi.
          </p>

          <p className="field-note">
            Thứ ra lò là một bản phối lập trình sạch sẽ, đúng nhịp đúng tông — không phải một bản
            mix ai đó ngồi một tuần. Hợp với hip-hop, lo-fi và những bài mà phần nền chỉ để đỡ
            giọng.
          </p>
        </>
      ) : generating ? (
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
