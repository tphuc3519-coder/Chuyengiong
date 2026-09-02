"""How a line is read out loud, as opposed to what reads it.

    text ──► split_blocks ──► plan() ──► [Beat, Beat, …] ──► engine + shape()

`tts.py` owns the engines; this module owns the reading. It exists to answer the
one complaint a fixed speaking rate and a fixed gap between sentences will
always earn: every sentence arrives at the same pace, the same loudness and the
same height, every pause is the same length whether it followed a comma or a
paragraph, and a page of that is a machine getting through a text rather than
somebody reading it out.

Nothing here is a second model, and nothing here guesses at what the text
*means*. It is the prosody that a synthesiser handed one sentence at a time
cannot see, applied from the outside, out of the parts of prosody that are
mechanical enough to be written down:

* **Pauses come from punctuation, and they are not all one length.** The
  published guidance for a phrase break is 120–300 ms and 400–700 ms for a
  paragraph or a dramatic one, so a comma, a full stop, an ellipsis and a blank
  line get four different silences instead of the single 0.25 s this used to
  insert everywhere. A segment the length budget cut mid-sentence gets the
  shortest of all, because there is no pause there in the writing.

* **Pitch declines across a paragraph and resets at the next one.** Declination
  is one of the most robust findings about connected speech, and its absence is
  precisely what makes independently synthesised sentences sound like a list.
  It is applied centred on zero — the first sentence a little above, the last a
  little below — so a paragraph's average pitch is unchanged and the F0
  measurement `pipeline._resolve_shift` runs afterwards reads what it always
  read.

* **A question is read higher, a statement falls.** The fall is the tail of
  the declination. The rise is the whole question sitting above its neighbours
  rather than a glide at the end of it: a final glide is the better imitation —
  yes/no questions being marked by a rising ending is about as settled as
  prosody gets — but drawing one means bending the pitch inside an utterance,
  and the only tool for that is a phase vocoder. See `_transpose` for what that
  did to Japanese. Overall F0 does go up on a question, so this is a real cue,
  just a quieter one.

* **Emotion is rate, pitch, range, loudness and pause length together.** The
  acoustic literature is consistent about the direction of each: happiness and
  anger raise F0, widen its range and speed the delivery up; sadness lowers and
  narrows F0, slows it down and puts more silence between the words. So a style
  here is five numbers in those directions rather than a label handed to a
  model that has never been trained on one — MMS-TTS and Kokoro both speak in
  one fixed voice with no emotion conditioning, and pretending otherwise would
  mean a third engine.

Two deliberate limits, because they are the reason this is small enough to
trust:

**It is sentence-level.** Emphasising one word means splitting a sentence in
the middle and synthesising the halves separately, which loses the
coarticulation across the join — a worse read, not a better one.

**Every deviation scales with one number.** `expressiveness` multiplies
everything below against the flat reading, so 0 is the plain delivery this used
to produce and 1.5 is as far as it goes. Only the deviations scale: the pause
after a comma is punctuation, not emotion, so its length survives at 0.

The DSP half (`shape`) needs numpy and librosa and imports them where it is
called, not at module scope: `pipeline` imports its way here on the small API
image, and the planning half is pure Python on purpose so the tests can cover
every rule above without a container.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- styles ---------------------------------------------------------------


@dataclass(frozen=True)
class Emotion:
    """One delivery, as the five things that actually differ between them.

    Multipliers are against the flat reading and offsets are against it too, so
    `natural` is every field at its identity and `expressiveness` has something
    to scale. `pitch_range` scales the per-sentence pitch moves rather than
    setting one of its own: a wider range is what the recordings show for the
    high-arousal emotions, and it is the difference between a happier voice and
    a voice that is merely higher.
    """

    label: str
    # Multiplier on the speaking rate the user asked for.
    rate: float = 1.0
    # Semitones the whole read sits above or below the synthesiser's register.
    pitch: float = 0.0
    gain_db: float = 0.0
    # Multiplier on every per-sentence pitch move: declination, the question
    # lift, the final fall.
    pitch_range: float = 1.0
    # Multiplier on every pause.
    pause: float = 1.0
    # VITS only, and multipliers on whatever the checkpoint's own config says
    # rather than absolute values — see `Beat.variation`.
    variation: float = 1.0
    duration_variation: float = 1.0


# The directions here are the ones the acoustic-emotion literature agrees on:
# happiness raises mean F0, widens its range, speeds the delivery up and is
# louder; sadness lowers and narrows F0, slows down, drops in level and leaves
# more silence between the words. The magnitudes are deliberately a fraction of
# the ones measured on acted speech — this is a reading voice, and the whole
# request was "a bit of emotion, not read carelessly", not a performance.
NATURAL = "natural"
EMOTIONS = {
    NATURAL: Emotion("Tự nhiên"),
    "warm": Emotion(
        "Ấm áp",
        rate=0.94,
        pitch=-0.4,
        gain_db=-0.5,
        pitch_range=0.90,
        pause=1.15,
        duration_variation=1.05,
    ),
    "cheerful": Emotion(
        "Vui vẻ",
        rate=1.08,
        pitch=1.2,
        gain_db=1.0,
        pitch_range=1.35,
        pause=0.85,
        variation=1.15,
        duration_variation=1.10,
    ),
    "sad": Emotion(
        "Trầm buồn",
        rate=0.86,
        pitch=-1.2,
        gain_db=-1.5,
        pitch_range=0.70,
        pause=1.40,
        variation=0.90,
        duration_variation=0.95,
    ),
    "serious": Emotion(
        "Nghiêm túc",
        rate=0.97,
        pitch=-0.5,
        gain_db=0.3,
        pitch_range=0.80,
        pause=1.05,
        variation=0.85,
        duration_variation=0.90,
    ),
}
DEFAULT_EMOTION = NATURAL

EXPRESSIVENESS_MIN = 0.0
EXPRESSIVENESS_MAX = 1.5
DEFAULT_EXPRESSIVENESS = 1.0


def resolve_emotion(name: str | None) -> Emotion:
    """The named style, or the natural one.

    Unknown falls back rather than raising, which is the opposite of what
    `tts.check_language` does with an unknown language and for the opposite
    reason: reading Vietnamese with the English checkpoint produces confident
    nonsense, while reading it without a style produces the reading this app
    shipped for months. A style is a slider, and a slider that arrives wrong is
    a client bug rather than something worth failing a paid GPU job over.
    """
    return EMOTIONS.get(name or "", EMOTIONS[DEFAULT_EMOTION])


def clean_emotion(name: str | None) -> str:
    """The id of the style that will be used, which is what the job records.

    `resolve_emotion` answers the same question with the style itself; this one
    exists so `/status` and the audit line say `natural` rather than repeating
    whatever the form field happened to hold.
    """
    return name if name in EMOTIONS else DEFAULT_EMOTION


def clamp_expressiveness(value: float | None) -> float:
    """How far every deviation below is taken. 0 is the flat read, 1 is normal."""
    try:
        depth = DEFAULT_EXPRESSIVENESS if value is None else float(value)
    except (TypeError, ValueError):
        return DEFAULT_EXPRESSIVENESS
    return max(EXPRESSIVENESS_MIN, min(EXPRESSIVENESS_MAX, depth))


def _cap(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _depth(depth: float, value: float, flat: float = 0.0) -> float:
    """`value` pulled back towards the flat reading by `depth`.

    `flat` is 0 for an offset and 1 for a multiplier; that is the only
    difference between the two kinds of number in this module.
    """
    return flat + (value - flat) * depth


# --- what closes a segment ------------------------------------------------

# Silence after a segment, in seconds, before the style's own multiplier.
#
# The numbers sit inside the published bands — 120–300 ms for a phrase break,
# 400–700 ms for a paragraph or a dramatic one — and the ordering between them
# is the part that matters: a comma is not a full stop, and a full stop is not a
# blank line.
RUN_ON = "run_on"
CLAUSE = "clause"
SENTENCE = "sentence"
QUESTION = "question"
EXCLAMATION = "exclamation"
TRAILING = "trailing"

PAUSE_SEC = {
    # No punctuation at all: the length budget cut a long sentence here, so
    # there is no pause in the writing and this is only the seam between two
    # synthesiser calls.
    RUN_ON: 0.12,
    CLAUSE: 0.20,
    SENTENCE: 0.40,
    QUESTION: 0.42,
    EXCLAMATION: 0.34,
    # An ellipsis is a pause that was written down on purpose.
    TRAILING: 0.60,
}
# A blank line, whatever closed the sentence before it.
PARAGRAPH_PAUSE_SEC = 0.75
# After the last segment there is nothing to pause for; `tts._join` pads both
# ends of the finished wav for the converter's sake and that is a different job.
FINAL_PAUSE_SEC = 0.0

_CLAUSE_MARKS = (",", ";", ":", "、", "，", "；", "：")
_QUESTION_MARKS = ("?", "？")
_EXCLAMATION_MARKS = ("!", "！")
_SENTENCE_MARKS = (".", "。")
_ELLIPSIS = ("…", "...", "。。。")
# Japanese asks a question with a final か and often no question mark at all —
# 「そうですか。」 is a question and ends in a full stop.
#
# Only ever read at a position where a sentence actually ends, and that
# condition is the whole point of it. か is an ordinary syllable inside ordinary
# words — しずか, たしか, なんとか, ほのか — and `_wrap` cuts a long Japanese
# sentence at an arbitrary character because the script gives it no spaces to
# cut at. Reading a fragment that merely stops after か as a question puts a
# question's intonation in the middle of a statement, which is the sort of wrong
# that a listener hears immediately.
_JA_QUESTION_TAILS = ("か", "かな", "かい", "かしら")
# Closing quotes and brackets sit outside the punctuation that classifies the
# sentence: 「本当ですか?」 and "Really?" end in a question either way.
_CLOSERS = "\"'”’」』）)]】》〉"


def classify(segment: str, sentence_end: bool = False) -> str:
    """What kind of break the end of `segment` is. See `PAUSE_SEC`.

    `sentence_end` says a sentence is known to end here even if nothing is
    written down to say so — the caller passes it for the last segment of a
    paragraph. It exists for Japanese: 「元気ですか」 typed without a full stop is
    a question, and 「きょうはとてもしずか…」 cut mid-sentence by the length
    budget is not, and the trailing か looks identical in both.
    """
    text = segment.rstrip().rstrip(_CLOSERS).rstrip()
    if not text:
        return RUN_ON
    if text.endswith(_ELLIPSIS):
        return TRAILING
    if text.endswith(_QUESTION_MARKS):
        return QUESTION
    if text.endswith(_EXCLAMATION_MARKS):
        return EXCLAMATION
    if text.endswith(_SENTENCE_MARKS):
        # 「そうですか。」 — a full stop closing a か is still a question.
        stem = text[:-1].rstrip()
        return QUESTION if stem.endswith(_JA_QUESTION_TAILS) else SENTENCE
    if text.endswith(_CLAUSE_MARKS):
        return CLAUSE
    # Nothing written down. A trailing か is the particle only where a sentence
    # really ends; anywhere else it is a syllable inside a word.
    if sentence_end and text.endswith(_JA_QUESTION_TAILS):
        return QUESTION
    return RUN_ON


# --- the rules ------------------------------------------------------------

# Total pitch travel across a paragraph, centred: the first sentence sits half
# of this above the middle and the last one half below, so the paragraph's mean
# is where it would have been. Declination is measured in far larger amounts
# than this in real speech; the point here is that it is not zero.
DECLINATION_ST = 1.2
# A question read higher than the sentences around it.
#
# This used to be a rise over the last third of a second, which is the better
# imitation of how a question is really read and is gone anyway — see
# `_transpose` for why nothing here bends the pitch inside a segment any more.
# A whole question sitting higher is the weaker cue of the two and the honest
# one to keep: overall F0 does go up on a question, and it costs a listener
# nothing because it rides the same clean transposition as everything else.
QUESTION_LIFT_ST = 1.0
QUESTION_RATE = 1.03
EXCLAMATION_LIFT_ST = 0.6
EXCLAMATION_GAIN_DB = 1.2
EXCLAMATION_RATE = 1.06
# Trailing off: quieter, slower, lower. Everything an ellipsis is for.
TRAILING_DROP_ST = 0.4
TRAILING_GAIN_DB = -1.5
TRAILING_RATE = 0.90
# The last statement of a paragraph falls further than declination alone.
FINAL_FALL_ST = 0.35
# The ceiling on a sentence's pitch move, once a wide style and a high
# `expressiveness` have both multiplied it.
#
# Transposing a whole sentence is safe for a tonal or a pitch-accent language —
# every ratio inside it is preserved, so Vietnamese tones and Japanese accent
# contours move along with it rather than being flattened. What the cap is
# actually for is the formants, which a resample moves with the pitch: past a
# couple of semitones the speaker starts to sound like a different size of
# person. The semitone shift the pipeline applies afterwards is capped at ±8 on
# a related argument (`audio_utils.MAX_SEMITONE_SHIFT`).
MAX_SENTENCE_PITCH_ST = 2.5
# …and is read slightly slower. Final lengthening, in the crudest form that
# still counts as having it.
FINAL_LENGTHENING = 0.96


@dataclass(frozen=True)
class Beat:
    """One segment and how to read it. Everything `plan` decides is here.

    `rate` is the pace a listener should hear. `synth_rate` is what the engine
    is actually asked for, and the two differ because of how the pitch is
    applied: `shape` transposes by resampling, which changes the length as well,
    so the engine is asked to speak by exactly the reciprocal and the length
    comes back. Feed `synth_rate` to the engine and `pitch` to `shape` and the
    result runs at `rate` — feed `rate` to the engine and every sentence with a
    pitch move comes out the wrong length.

    `gain_db` is applied to the audio too. `variation` and `duration_variation`
    are multipliers on the checkpoint's own `noise_scale` and
    `noise_scale_duration` rather than values, since those defaults are per
    checkpoint and overwriting them with a constant would be a change no style
    asked for.
    """

    text: str
    kind: str
    rate: float = 1.0
    synth_rate: float = 1.0
    pitch: float = 0.0
    gain_db: float = 0.0
    pause_sec: float = 0.0
    variation: float = 1.0
    duration_variation: float = 1.0


def plan(
    blocks: list[list[str]],
    *,
    emotion: str = DEFAULT_EMOTION,
    speaking_rate: float = 1.0,
    expressiveness: float = DEFAULT_EXPRESSIVENESS,
) -> list[Beat]:
    """A `Beat` per segment: how fast, how high, how loud, and how long after.

    `blocks` is `tts.split_blocks` output — paragraphs of segments — because
    two of the rules are about where a paragraph starts and ends, and a flat
    list of sentences has thrown that away.
    """
    style = resolve_emotion(emotion)
    depth = clamp_expressiveness(expressiveness)
    # Every style number, pulled back towards flat once rather than at each use.
    style_rate = _depth(depth, style.rate, 1.0)
    style_pitch = _depth(depth, style.pitch)
    style_gain = _depth(depth, style.gain_db)
    style_range = _depth(depth, style.pitch_range, 1.0) * depth
    style_pause = _depth(depth, style.pause, 1.0)
    variation = _depth(depth, style.variation, 1.0)
    duration_variation = _depth(depth, style.duration_variation, 1.0)

    # Dropped before the loop, not skipped inside it: an empty paragraph at the
    # end would otherwise leave the last real sentence believing something
    # follows it, and close the recording on three quarters of a second of
    # silence waiting for a paragraph that never comes.
    blocks = [block for block in blocks if block]

    beats: list[Beat] = []
    for block_index, block in enumerate(blocks):
        last_block = block_index == len(blocks) - 1
        count = len(block)
        for index, segment in enumerate(block):
            last_in_block = index == count - 1
            # A paragraph ends where its last segment does, whether or not the
            # writer put a full stop there. Japanese needs to be told.
            kind = classify(segment, sentence_end=last_in_block)
            # Centred declination: +half a span at the top of the paragraph,
            # -half at the bottom, nothing at all in a one-sentence paragraph.
            span = index / (count - 1) if count > 1 else 0.5
            pitch = DECLINATION_ST * (0.5 - span)
            rate = 1.0
            gain_db = 0.0

            if kind == QUESTION:
                pitch += QUESTION_LIFT_ST
                rate *= QUESTION_RATE
            elif kind == EXCLAMATION:
                pitch += EXCLAMATION_LIFT_ST
                gain_db += EXCLAMATION_GAIN_DB
                rate *= EXCLAMATION_RATE
            elif kind == TRAILING:
                pitch -= TRAILING_DROP_ST
                gain_db += TRAILING_GAIN_DB
                rate *= TRAILING_RATE
            elif kind == SENTENCE and last_in_block:
                # A paragraph ends lower than its own declination would put it.
                pitch -= FINAL_FALL_ST

            if last_in_block:
                rate *= FINAL_LENGTHENING

            if last_in_block and last_block:
                pause = FINAL_PAUSE_SEC
            elif last_in_block:
                pause = max(PAUSE_SEC[kind], PARAGRAPH_PAUSE_SEC)
            else:
                pause = PAUSE_SEC[kind]

            # The style's own offset moves the whole read; the sentence's
            # move is what `pitch_range` widens or narrows.
            moved = _cap(style_pitch + pitch * style_range, MAX_SENTENCE_PITCH_ST)
            heard = speaking_rate * style_rate * _depth(depth, rate, 1.0)
            beats.append(
                Beat(
                    text=segment,
                    kind=kind,
                    rate=heard,
                    # Asked for slower by exactly what the transposition will
                    # speed it up by, so the sentence lands back on `heard`.
                    synth_rate=heard / semitone_ratio(moved),
                    pitch=moved,
                    gain_db=style_gain + _depth(depth, gain_db),
                    pause_sec=pause * style_pause,
                    variation=variation,
                    duration_variation=duration_variation,
                )
            )
    return beats


# --- applying it to the audio ---------------------------------------------

# Below this a pitch move is not worth resampling for — it is under the
# just-noticeable difference for a shift of a whole utterance either way.
MIN_AUDIBLE_ST = 0.15


def semitone_ratio(semitones: float) -> float:
    """The frequency ratio `semitones` is, which is also the length ratio.

    One number does both jobs because the transposition is a resample: raising
    a sentence by `s` semitones multiplies its pitch by this and divides its
    length by the same, which is what `Beat.synth_rate` exists to undo.
    """
    return 2.0 ** (semitones / 12.0)


def _transpose(audio, sample_rate: int, semitones: float):
    """`audio` transposed by resampling it. Unchanged if librosa is missing.

    **A resample, and deliberately not a phase vocoder.** The obvious call here
    is `librosa.effects.pitch_shift`, which time-stretches with a phase vocoder
    and then resamples so the length comes out unchanged. It was the first thing
    this module did and it was wrong, in a way that was worst for the language
    with the least room for it:

    a phase vocoder analyses in windows — librosa's default is 2048 samples,
    which is 85 ms at Kokoro's 24 kHz — and it smears anything shorter than a
    window across it. Japanese spends its meaning on exactly that scale. っ is a
    stop of silence, ー is a held vowel, ん is a mora of its own, and the
    difference between きて and きって is how long a gap is. Smear those and the
    words change. It ran on every sentence, for a declination move of well under
    a semitone that nobody would have noticed missing.

    Resampling has no window and no analysis. It is a tape-speed change: every
    ratio in the signal is preserved exactly, so a pitch-accent contour, a
    Vietnamese tone and a geminate consonant all survive it intact. What it does
    instead is change the length — which is why nothing calls this without
    `Beat.synth_rate` having already paid for it — and shift the formants with
    the pitch, which is why `MAX_SENTENCE_PITCH_ST` is small and why it matters
    little here anyway: Seed-VC replaces the timbre one step later.

    librosa is in `base_image` and therefore in both synthesiser images. The
    fallback is for the unit tests, which run on a bare CI box.
    """
    import numpy as np

    try:
        import librosa
    except ImportError:  # pragma: no cover - librosa is present in the images
        return audio
    # Reading the samples as if they had been recorded slower and playing them
    # at the original rate: pitch up by `ratio`, length down by it.
    target = int(round(sample_rate / semitone_ratio(semitones)))
    if target <= 0 or target == sample_rate:
        return audio
    moved = librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=sample_rate,
        target_sr=target,
        res_type="soxr_hq",
    )
    return np.asarray(moved, dtype=np.float32)


def shape(audio, sample_rate: int, beat: Beat):
    """One synthesised segment, read the way its `Beat` says.

    Level, then the transposition. Nothing here bends the pitch *inside* a
    segment: a contour drawn over part of an utterance cannot be done by
    resampling, and the only other way to draw one is the vocoder `_transpose`
    exists to avoid. A question is read higher rather than rising at the end,
    which is the weaker cue and the one that does not damage the words.

    Not clipped: `tts._join` normalises the finished wav to a fixed peak, and
    clipping here would throw away the headroom it is about to use.
    """
    import numpy as np

    shaped = np.asarray(audio, dtype=np.float32)
    if abs(beat.gain_db) > 0.01:
        shaped = shaped * np.float32(10.0 ** (beat.gain_db / 20.0))
    if abs(beat.pitch) >= MIN_AUDIBLE_ST:
        shaped = _transpose(shaped, sample_rate, beat.pitch)
    return shaped
