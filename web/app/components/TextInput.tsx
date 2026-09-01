"use client";

import { useId } from "react";

import { LANGUAGES, maxCharsFor } from "@/lib/params";

/**
 * What the `tts` branch has instead of a file: a box to write in, and the
 * language to read it in.
 *
 * The language sits here rather than under "Tinh chỉnh" because it is not a
 * refinement — the checkpoint that reads Vietnamese cannot read German, so
 * picking the wrong one produces a fluent recording of nothing. It belongs
 * beside the text it applies to.
 *
 * The counter turns into a warning before the limit rather than at it: the
 * backend refuses text past the limit outright, and finding that out after
 * writing three paragraphs and uploading a voice sample is a bad way to learn
 * it. `maxLength` stops the typing; the counter explains why.
 *
 * The limit moves with the language — 700 characters of Japanese is as much
 * speech as 2000 of Vietnamese — so switching language can leave text already
 * over it. `maxLength` cannot take words back, so the counter goes red and
 * says how many are past; the page keeps the button disabled until they are.
 */
export function TextInput({
  text,
  onText,
  language,
  onLanguage,
  disabled,
}: {
  text: string;
  onText: (text: string) => void;
  language: string;
  onLanguage: (language: string) => void;
  disabled?: boolean;
}) {
  const textId = useId();
  const languageId = useId();
  const limit = maxCharsFor(language);
  const left = limit - text.length;
  const tight = left <= limit / 10;

  return (
    <div className="composer">
      <label className="visually-hidden" htmlFor={textId}>
        Văn bản cần đọc
      </label>
      <textarea
        id={textId}
        className="composer-text"
        value={text}
        rows={6}
        maxLength={limit}
        disabled={disabled}
        placeholder="Gõ hoặc dán đoạn văn bản muốn nghe bằng giọng mẫu…"
        onChange={(event) => onText(event.target.value)}
      />

      <div className="composer-foot">
        <label className="composer-language" htmlFor={languageId}>
          <span>Ngôn ngữ</span>
          <select
            id={languageId}
            value={language}
            disabled={disabled}
            onChange={(event) => onLanguage(event.target.value)}
          >
            {LANGUAGES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <span className={tight ? "composer-count is-tight" : "composer-count"}>
          {left < 0 ? `thừa ${-left} ký tự` : `còn ${left} ký tự`}
        </span>
      </div>

      {/*
        Not a footnote: the model tokenises characters against a per-language
        vocabulary that has letters and punctuation in it and no digits, so a
        number is dropped silently — "25 tuổi" is read as "tuổi". Saying so
        here is cheaper than every user discovering it once.
      */}
      <p className="field-note">
        Viết số và ký hiệu thành chữ — “25” hay “%” sẽ bị bỏ qua khi đọc. Chấm câu thì giữ nguyên,
        nó là chỗ ngắt nghỉ.
      </p>
    </div>
  );
}
