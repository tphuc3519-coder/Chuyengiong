"use client";

import { useId } from "react";

import { FileDrop } from "./FileDrop";
import {
  AUDIO_ACCEPT,
  BEAT_PROMPT_CHARS,
  BEAT_PROMPT_EXAMPLES,
  BEAT_RANDOM_SEED,
  MAX_INPUT_BYTES,
  type BeatSource as Source,
  type Params,
} from "@/lib/params";

/**
 * Where the replacement backing track comes from, on the beat branches.
 *
 * Three sources and exactly one of them, so this is a radio group rather than
 * three independent fields — and where the deployment does not ship the
 * generator only `upload` exists, so the group disappears entirely rather than
 * rendering a choice of one.
 *
 * **"Phối lại bài này" is first because it is what the mode is called.** The
 * question this control answers, for somebody who came here to change a beat,
 * is "does the app make one from my song or do I bring one" — and for a long
 * time the honest answer was "you bring one", which read as the feature not
 * existing. Deriving is now the default and the other two are the ways out of
 * it: a description when the song's own harmony is not wanted, and a file when
 * a person has already made the arrangement.
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
  const source: Source = canGenerate ? params.beatSource : "upload";
  const left = BEAT_PROMPT_CHARS - params.beatPrompt.length;

  const options: { value: Source; label: string; hint: string }[] = [
    {
      value: "derive",
      label: "Phối lại bài này",
      hint: "App đọc hợp âm của bài rồi dựng bản phối mới cùng tông",
    },
    {
      value: "generate",
      label: "Beat mới từ mô tả",
      hint: "Bạn tả kiểu nhạc, app sinh beat không liên quan bài gốc",
    },
    { value: "upload", label: "Tự đưa beat", hint: "Beat bạn đã có sẵn quyền dùng" },
  ];

  return (
    <div>
      {canGenerate && (
        <>
          <span className="slider-label" id={groupId}>
            Beat mới lấy từ đâu
          </span>
          <div className="segmented segmented-stack" role="radiogroup" aria-labelledby={groupId}>
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={source === option.value}
                className={source === option.value ? "segment is-active" : "segment"}
                disabled={disabled}
                onClick={() => onChange({ ...params, beatSource: option.value })}
              >
                <span className="segment-label">{option.label}</span>
                <span className="segment-hint">{option.hint}</span>
              </button>
            ))}
          </div>
        </>
      )}

      {source === "derive" && (
        <>
          <p className="field-note">
            App tách nhạc nền ra, đo tốc độ và tông, <strong>dò vòng hợp âm</strong> của bài, rồi
            bảo máy đánh lại đúng vòng hợp âm đó bằng nhạc cụ khác. Beat ra vẫn đi theo bài — kể cả
            ở những ô nhịp đổi hợp âm, chỗ mà beat sinh từ mô tả sẽ chỏi.
          </p>

          <label className="slider" htmlFor={promptId}>
            <span className="slider-label">
              Muốn nghe ra kiểu gì <output>còn {left} ký tự</output>
            </span>
            <textarea
              id={promptId}
              className="composer-text"
              rows={2}
              maxLength={BEAT_PROMPT_CHARS}
              value={params.beatPrompt}
              disabled={disabled}
              placeholder="rock, guitar méo, trống thật — để trống cũng được"
              onChange={(event) => onChange({ ...params, beatPrompt: event.target.value })}
            />
            <span className="slider-hint">
              Chỗ này chỉ quyết định <em>nhạc cụ và chất nhạc</em>; hợp âm và tốc độ đã lấy từ bài
              rồi. Để trống thì app tự viết mô tả từ tông và BPM đo được.
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

          {/*
            The one control on this page that is a legal question, so it gets a
            sentence rather than a label. Unchecked is the branch that copies
            nothing, and the wording has to make clear that ticking it is a
            trade rather than a quality setting.
          */}
          <label className="checkbox">
            <input
              type="checkbox"
              checked={params.beatInit === "original"}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...params, beatInit: event.target.checked ? "original" : "sketch" })
              }
            />
            <span>Cho máy nghe thẳng nhạc nền gốc (bám bài sát hơn)</span>
          </label>
          <p className="field-note">
            Mặc định máy <strong>không</strong> nghe bản ghi của bạn: nó chỉ nhận vòng hợp âm do app
            tự đánh lại, nên beat ra là bản cover phần sáng tác — thứ xin phép được. Bật ô trên thì
            máy nghe thẳng nhạc nền gốc: giống bài hơn, nhưng sản phẩm khi đó là tác phẩm{" "}
            <strong>phái sinh của chính bản ghi</strong>, tức là đúng thứ mà mục này sinh ra để
            tránh. Bật khi bạn có quyền với bản ghi, hoặc chấp nhận rủi ro đó.
          </p>
        </>
      )}

      {source === "generate" && (
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
            Beat sinh theo mô tả <strong>không biết vòng hợp âm</strong> của bài, nên nó hợp với
            rap, hip-hop và nhạc điện tử — bài nào giọng đi giai điệu nhiều thì dễ chỏi ở những ô
            nhịp đổi hợp âm. Muốn beat đi theo hợp âm của bài thì chọn “Phối lại bài này”.
          </p>
        </>
      )}

      {source === "upload" && (
        <>
          {/*
            Nói trước khi hỏi. Người dùng vừa bấm một nút tên "Đổi beat" nên rất
            dễ chờ app tự làm ra beat; câu này là chỗ duy nhất đính chính điều
            đó, và nó phải nằm *trên* ô chọn file chứ không phải dưới.
          */}
          <p className="field-note">
            Bạn đưa beat mới vào đây, app sẽ đo bài gốc rồi kéo beat cho khớp và ghép giọng lên.
            Muốn app tự làm ra beat thì chọn “Phối lại bài này” ở trên — nếu không thấy lựa chọn đó
            thì bản triển khai này chưa bật phần sinh beat.
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
