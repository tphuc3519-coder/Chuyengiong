"use client";

import { useId } from "react";

import { BEAT_STYLES, type Params } from "@/lib/params";

/**
 * Which kind of music the new beat is.
 *
 * The control this whole mode was missing. "Đổi beat" without it asks the user
 * to describe music in a text box — which is a fine thing to be able to do and
 * a bad thing to be *required* to do, because most people who want a lo-fi beat
 * want a lo-fi beat and have no interest in writing the words "soft
 * tape-saturated drums".
 *
 * **Shown for every source, including "Tự đưa beat".** Half of a style is the
 * description handed to the generator and half is the mix — how far the bed
 * ducks under a syllable, where it is carved out for the consonants, how much
 * bottom end it keeps. The second half applies to a beat somebody uploaded
 * exactly as much as to one the app made, and it is the half that decides
 * whether the result sounds like a record or like two files playing at once.
 *
 * A grid of buttons rather than a `<select>`: twelve options each of which
 * needs a line of explanation is not a dropdown, and on a phone a dropdown
 * hides eleven of them behind a tap.
 */
export function BeatStyle({
  params,
  onChange,
  disabled,
  /** What the hint under the group should say for the source in use. */
  note,
}: {
  params: Params;
  onChange: (params: Params) => void;
  disabled?: boolean;
  note: string;
}) {
  const groupId = useId();

  return (
    <div className="style-picker">
      <span className="slider-label" id={groupId}>
        Kiểu nhạc
      </span>
      <div className="style-grid" role="radiogroup" aria-labelledby={groupId}>
        {BEAT_STYLES.map((style) => (
          <button
            key={style.id}
            type="button"
            role="radio"
            aria-checked={params.beatStyle === style.id}
            title={style.hint}
            className={params.beatStyle === style.id ? "style-chip is-active" : "style-chip"}
            disabled={disabled}
            onClick={() => onChange({ ...params, beatStyle: style.id })}
          >
            <span className="segment-label">{style.label}</span>
            <span className="segment-hint">{style.hint}</span>
          </button>
        ))}
      </div>
      <span className="slider-hint">{note}</span>
    </div>
  );
}
