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
 * Where the replacement backing track comes from, on the beat branches.
 *
 * Two sources and exactly one of them, so this is a radio group rather than two
 * independent fields — and where the deployment does not ship the generator
 * there is only one source, so the group disappears entirely rather than
 * rendering a choice of one.
 *
 * Upload is the default because it is the one that always works: no GPU, no
 * gated weights, and the licence of what comes out is whatever licence the user
 * already had. It is also the only one that reaches a real arrangement — a beat
 * somebody made is a beat somebody made.
 *
 * There was a third, "phối lại bài này", which read the song's own chords and
 * played them back on synthesised instruments. It is gone. Held against a human
 * rock arrangement it put 55% of its energy below 120 Hz and had nothing above
 * 4 kHz, and that gap is not a tuning problem — it is the distance between
 * generating waveforms from arithmetic and a sampled instrument library.
 */
export function BeatSource({
  params,
  beat,
  onBeat,
  onChange,
  canGenerate,
  disabled,
}: {
  params: Params;
  beat: File | null;
  onBeat: (file: File | null) => void;
  onChange: (params: Params) => void;
  /** Whether this deployment ships the generator — asked at run time. */
  canGenerate: boolean;
  disabled?: boolean;
}) {
  const groupId = useId();
  const promptId = useId();
  const generating = canGenerate && params.beatSource === "generate";
  const left = BEAT_PROMPT_CHARS - params.beatPrompt.length;

  return (
    <div>
      {canGenerate && (
        <>
          <span className="slider-label" id={groupId}>
            Beat mới lấy từ đâu
          </span>
          <div className="segmented" role="radiogroup" aria-labelledby={groupId}>
            <button
              type="button"
              role="radio"
              aria-checked={generating}
              className={generating ? "segment is-active" : "segment"}
              disabled={disabled}
              onClick={() => onChange({ ...params, beatSource: "generate" })}
            >
              <span className="segment-label">Máy làm beat</span>
              <span className="segment-hint">Mô tả kiểu nhạc, app tự sinh beat mới</span>
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={!generating}
              className={generating ? "segment" : "segment is-active"}
              disabled={disabled}
              onClick={() => onChange({ ...params, beatSource: "upload" })}
            >
              <span className="segment-label">Tự đưa beat</span>
              <span className="segment-hint">Beat bạn đã có sẵn quyền dùng</span>
            </button>
          </div>
        </>
      )}

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
        <>
          {/*
            Nói trước khi hỏi. Người dùng vừa bấm một nút tên "Đổi beat" nên rất
            dễ chờ app tự làm ra beat; câu này là chỗ duy nhất đính chính điều
            đó, và nó phải nằm *trên* ô chọn file chứ không phải dưới.
          */}
          <p className="field-note">
            Bạn đưa beat mới vào đây, app sẽ đo bài gốc rồi kéo beat cho khớp và ghép giọng lên.
            Muốn app tự làm ra beat thì chọn “Máy làm beat” ở trên — nếu không thấy lựa chọn đó thì
            bản triển khai này chưa bật phần sinh beat.
          </p>
          <FileDrop
            file={beat}
            onFile={onBeat}
            accept={AUDIO_ACCEPT}
            maxBytes={MAX_INPUT_BYTES}
            label="Beat mới của bạn"
            hint="Kéo thả hoặc bấm để chọn · beat instrumental, không có lời"
            disabled={disabled}
          />
          <p className="field-note">
            Beat được đo BPM và tông rồi cắt tròn ô nhịp, dịch tông, kéo tempo và lặp cho khớp bài —
            nên nó <strong>không cần</strong> cùng tốc độ hay cùng tông với bản gốc. Đổi lại, chất
            lượng bản phối chính là chất lượng file bạn đưa vào: phần này khớp nhạc, nó không sáng
            tác.
          </p>
        </>
      )}
    </div>
  );
}
