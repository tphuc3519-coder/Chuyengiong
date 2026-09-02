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

* **A question rises, a statement falls.** The fall is the tail of the
  declination; the rise is a short glide over the last third of a second, which
  is the one piece of intonation a listener notices immediately when it is
  missing. Yes/no question intonation being marked by a rising ending, and a
  declarative by a falling one, is about as settled as prosody gets.

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
    # rise, the final fall.
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
# 「そうですか。」 is a question and ends in a full stop. Only the sentence-final
# position is read, so the か that means "or" in the middle of one is untouched.
_JA_QUESTION_TAILS = ("か", "かな", "かい", "かしら")
# Closing quotes and brackets sit outside the punctuation that classifies the
# sentence: 「本当ですか?」 and "Really?" end in a question either way.
_CLOSERS = "\"'”’」』）)]】》〉"


def classify(segment: str) -> str:
    """What kind of break the end of `segment` is. See `PAUSE_SEC`."""
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
    if text.endswith(_JA_QUESTION_TAILS):
        return QUESTION
    return RUN_ON


# --- the rules ------------------------------------------------------------

# Total pitch travel across a paragraph, centred: the first sentence sits half
# of this above the middle and the last one half below, so the paragraph's mean
# is where it would have been. Declination is measured in far larger amounts
# than this in real speech; the point here is that it is not zero.
DECLINATION_ST = 1.2
# The final rise on a question, over `RISE_SEC` at the end of the segment.
QUESTION_RISE_ST = 2.2
# …and the whole question sitting slightly higher, which is the other half of
# how one is read.
QUESTION_LIFT_ST = 0.4
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
# Ceilings on what the two pitch numbers can reach once a wide style and a high
# `expressiveness` have both multiplied them.
#
# These exist for Vietnamese, and for every other tonal language this will read.
# Transposing a whole sentence is safe — the tone contours move with it — but the
# final rise is a contour of its own laid over the last syllable, and a large one
# is the difference between asking a question and saying a different word. The
# semitone shift the pipeline applies afterwards is capped at ±8 for the same
# reason (`audio_utils.MAX_SEMITONE_SHIFT`); this is that argument at the scale
# of one sentence.
MAX_SENTENCE_PITCH_ST = 2.5
MAX_RISE_ST = 3.5
# …and is read slightly slower. Final lengthening, in the crudest form that
# still counts as having it.
FINAL_LENGTHENING = 0.96


@dataclass(frozen=True)
class Beat:
    """One segment and how to read it. Everything `plan` decides is here.

    `rate` is absolute, ready for the engine. `pitch`, `gain_db` and `rise` are
    applied to the audio afterwards by `shape`, because neither engine takes
    them: MMS exposes duration and two noise scales and nothing else, and
    Kokoro exposes speed. `variation` and `duration_variation` are multipliers
    on the checkpoint's own `noise_scale` and `noise_scale_duration` rather than
    values, since those defaults are per checkpoint and overwriting them with a
    constant would be a change no style asked for.
    """

    text: str
    kind: str
    rate: float = 1.0
    pitch: float = 0.0
    gain_db: float = 0.0
    rise: float = 0.0
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
            kind = classify(segment)
            last_in_block = index == count - 1
            # Centred declination: +half a span at the top of the paragraph,
            # -half at the bottom, nothing at all in a one-sentence paragraph.
            span = index / (count - 1) if count > 1 else 0.5
            pitch = DECLINATION_ST * (0.5 - span)
            rate = 1.0
            gain_db = 0.0
            rise = 0.0

            if kind == QUESTION:
                pitch += QUESTION_LIFT_ST
                rise = QUESTION_RISE_ST
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

            beats.append(
                Beat(
                    text=segment,
                    kind=kind,
                    rate=speaking_rate * style_rate * _depth(depth, rate, 1.0),
                    # The style's own offset moves the whole read; the
                    # sentence's move is what `pitch_range` widens or narrows.
                    pitch=_cap(style_pitch + pitch * style_range, MAX_SENTENCE_PITCH_ST),
                    gain_db=style_gain + _depth(depth, gain_db),
                    rise=_cap(rise * style_range, MAX_RISE_ST),
                    pause_sec=pause * style_pause,
                    variation=variation,
                    duration_variation=duration_variation,
                )
            )
    return beats


# --- applying it to the audio ---------------------------------------------

# Below this a pitch move is not worth a phase vocoder pass — it is under the
# just-noticeable difference for a shift of a whole utterance either way.
MIN_AUDIBLE_ST = 0.15
# The final rise happens over the last third of a second, which is roughly the
# last stressed syllable and whatever follows it.
RISE_SEC = 0.35
# The glide as a staircase. Four steps of a 2.2 semitone rise is 0.55 apiece
# across 350 ms, which reads as a glide rather than as four notes; a real
# time-varying resample would need a vocoder of our own.
RISE_STEPS = 4
# Equal-power crossfade between the steps, through `audio_utils.crossfade_concat`
# — the same join the converter's chunks already use.
RISE_OVERLAP_SEC = 0.02
# STFT window for the transposition. librosa's own default, and the floor
# `_window` stops halving at: 256 samples is 16 ms at 16 kHz, which still holds
# two periods of a 120 Hz voice.
DEFAULT_WINDOW = 2048
MIN_WINDOW = 256


def _shift(audio, sample_rate: int, semitones: float):
    """`audio` transposed, same length. Unchanged if librosa is not installed.

    librosa is in `base_image` and therefore in both synthesiser images. The
    fallback is for the unit tests, which run on a bare CI box: everything in
    `plan` is testable there, and a missing pitch move is the right way for the
    part that is not to stay out of the way.
    """
    import numpy as np

    try:
        import librosa
    except ImportError:  # pragma: no cover - librosa is present in the images
        return audio
    shifted = librosa.effects.pitch_shift(
        y=audio, sr=sample_rate, n_steps=float(semitones), n_fft=_window(len(audio))
    )
    return np.asarray(shifted, dtype=np.float32)


def _window(length: int) -> int:
    """The STFT window to transpose `length` samples with.

    librosa's default is 2048, which is longer than a glide step (~90 ms, so
    1400 samples at 16 kHz) and longer than a very short segment. A window
    longer than the signal is padding rather than analysis, and librosa says so
    on stderr every time. So halve it until it fits, with a floor low enough to
    still hold a couple of periods of a low voice.
    """
    window = DEFAULT_WINDOW
    while window > MIN_WINDOW and window > length:
        window //= 2
    return window


def _glide(audio, sample_rate: int, semitones: float):
    """The last `RISE_SEC` bent upward by `semitones`, in `RISE_STEPS` steps."""
    import numpy as np

    from .audio_utils import crossfade_concat

    tail_len = int(RISE_SEC * sample_rate)
    step = tail_len // RISE_STEPS
    overlap = int(RISE_OVERLAP_SEC * sample_rate)
    # Nothing to bend: a segment barely longer than the tail would have its
    # whole self transposed, which is not a final rise.
    if len(audio) < tail_len * 2 or step <= overlap * 2:
        return audio

    head, tail = audio[:-tail_len], audio[-tail_len:]
    pieces = []
    for i in range(RISE_STEPS):
        start = i * step
        stop = tail_len if i == RISE_STEPS - 1 else start + step
        # Each piece but the first reaches back one overlap, which is the
        # contract `crossfade_concat` reassembles from.
        begin = start if i == 0 else start - overlap
        pieces.append(_shift(tail[begin:stop], sample_rate, semitones * (i + 1) / RISE_STEPS))
    return np.concatenate([head, crossfade_concat(pieces, sample_rate, RISE_OVERLAP_SEC)])


def shape(audio, sample_rate: int, beat: Beat):
    """One synthesised segment, read the way its `Beat` says.

    Level, then the whole-segment transposition, then the final rise. Not
    clipped: `tts._join` normalises the finished wav to a fixed peak, and
    clipping here would throw away the headroom it is about to use.
    """
    import numpy as np

    shaped = np.asarray(audio, dtype=np.float32)
    if abs(beat.gain_db) > 0.01:
        shaped = shaped * np.float32(10.0 ** (beat.gain_db / 20.0))
    if abs(beat.pitch) >= MIN_AUDIBLE_ST:
        shaped = _shift(shaped, sample_rate, beat.pitch)
    if beat.rise >= MIN_AUDIBLE_ST:
        shaped = _glide(shaped, sample_rate, beat.rise)
    return shaped
