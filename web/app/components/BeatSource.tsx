"use client";

import { useId } from "react";

import { BeatStyle } from "./BeatStyle";
import { FileDrop } from "./FileDrop";
import {
  AUDIO_ACCEPT,
  BEAT_FOLLOW_MAX,
  BEAT_FOLLOW_MIN,
  BEAT_PROMPT_CHARS,
  BEAT_PROMPT_EXAMPLES,
  BEAT_RANDOM_SEED,
  MAX_INPUT_BYTES,
  formatPercent,
  type BeatSource as Source,
  type Params,
} from "@/lib/params";

/**
 * Where the slider sits before anybody has moved it.
 *
 * Purely cosmetic: while `beatFollow` is null the backend uses the default for
 * whichever init source is in use, and this is only the thumb's position. It is
 * the midpoint of the two backend defaults so the thumb does not jump when the
 * checkbox below is ticked.
 */
const DEFAULT_FOLLOW_POSITION = 0.4;

/**
 * Where the replacement backing track comes from, on the beat branches.
 *
 * Three sources and exactly one of them, so this is a radio group rather than
 * three independent fields — and where the deployment does not ship the
 * generator only `upload` exists, so the group disappears entirely rather than
 * rendering a choice of one.
 *
 * The **style** picker sits above the three, because it is the question a
 * person actually came here with. "Đổi beat sang lo-fi" is one thought; "where
 * does the audio come from" is a follow-up, and for two of the three sources
 * the style is most of the answer to it. It is rendered for all three,
 * including `upload` — see `BeatStyle` for why a style still applies to a beat
 * somebody brought themselves.
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
  const followId = useId();
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

          <BeatStyle
            params={params}
            onChange={onChange}
            disabled={disabled}
            note="Chọn kiểu là đủ — không cần gõ gì thêm. Hợp âm, tông và tốc độ đã lấy từ bài; chỗ này chỉ quyết định nhạc cụ, và cách beat nhường chỗ cho giọng khi hát."
          />

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
              Chỉ dùng khi bạn muốn nói cụ thể hơn kiểu đã chọn ở trên — gõ vào đây là{" "}
              <strong>thay</strong> mô tả của kiểu đó, không phải cộng thêm. Để trống là dùng đúng
              kiểu đã chọn.
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
            <span>Cho máy nghe thẳng nhạc nền gốc (beat ra sẽ rất giống bản gốc)</span>
          </label>
          <p className="field-note">
            Mặc định máy <strong>không</strong> nghe bản ghi của bạn: nó chỉ nhận vòng hợp âm do app
            tự đánh lại, nên beat ra là <strong>một bản phối mới</strong> của phần sáng tác — khác
            bản gốc rõ rệt, và là thứ xin phép được.
          </p>

          {/*
            The dial the two ends of this feature turned out to need. Reported
            by ear: starting from the recording at 0.65 gave back something
            barely different from the original, and starting from the chord
            sketch at 0.35 sounded bad. Those are two different sources rather
            than two points on one axis, so no third number could be guessed
            from them — and each guess would have cost a deploy and a listen.
            "Tự động" is the untouched position and changes nothing.
          */}
          <label className="slider" htmlFor={followId}>
            <span className="slider-label">
              Bám bài bao nhiêu
              <output>
                {params.beatFollow === null ? "Tự động" : formatPercent(params.beatFollow)}
              </output>
            </span>
            <input
              id={followId}
              type="range"
              min={BEAT_FOLLOW_MIN}
              max={BEAT_FOLLOW_MAX}
              step={0.05}
              value={params.beatFollow ?? DEFAULT_FOLLOW_POSITION}
              disabled={disabled}
              onChange={(event) => onChange({ ...params, beatFollow: Number(event.target.value) })}
            />
            <span className="slider-hint">
              Kéo thấp: máy tự do hơn, beat hay hơn nhưng dễ đi lạc khỏi bài. Kéo cao: bám bài sát
              hơn, nhưng lên quá thì beat ra <strong>gần giống thứ máy được nghe</strong> — và với ô
              bên dưới bật thì thứ đó chính là bản gốc.{" "}
              {params.beatFollow !== null && (
                <button
                  type="button"
                  className="linkish"
                  disabled={disabled}
                  onClick={() => onChange({ ...params, beatFollow: null })}
                >
                  Về tự động
                </button>
              )}
            </span>
          </label>
          <p className="field-note">
            Bật ô trên thì máy nghe thẳng nhạc nền gốc và viết đè lên đó. Hai hệ quả, và cả hai đều
            là lý do để cân nhắc: beat ra <strong>nghe gần giống bản gốc</strong> — nếu bạn muốn một
            bản phối khác đi thì đây là ô cần tắt chứ không phải ô cần bật — và sản phẩm khi đó là
            tác phẩm <strong>phái sinh của chính bản ghi</strong>, tức là đúng thứ mà mục này sinh
            ra để tránh. Bật khi bạn có quyền với bản ghi, hoặc chấp nhận rủi ro đó.
          </p>
        </>
      )}

      {source === "generate" && (
        <>
          <BeatStyle
            params={params}
            onChange={onChange}
            disabled={disabled}
            note="Chọn một kiểu là đã đủ để bấm chạy. Ô mô tả bên dưới chỉ cần khi bạn muốn nói cụ thể hơn."
          />

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
              Gõ vào đây là <strong>thay</strong> mô tả của kiểu đã chọn, không phải cộng thêm.
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

          <BeatStyle
            params={params}
            onChange={onChange}
            disabled={disabled}
            note="Beat là của bạn nên phần mô tả không dùng đến — kiểu chọn ở đây quyết định cách phối: beat nhường chỗ cho giọng nhiều hay ít, khoét ở dải nào, giữ lại bao nhiêu tiếng trầm."
          />
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
