"""Making one piece of music sit under another one.

    beat ──► plan_fit(beat, bài) ──► Fit(dịch bao nhiêu, kéo bao nhiêu)
         └─► fit() ──► cắt tròn ô nhịp ──► dịch tone ──► kéo tempo
                   ──► lặp cho đủ dài ──► đẩy vào đúng phách ──► wav

`analysis.py` measures; this module acts on the measurements. Between them they
are the whole answer to "đổi beat" — and the reason the answer is two modules is
that measuring is arithmetic anybody can check, while acting on it is four
ffmpeg filters whose order matters.

Four things happen, and each is one decision:

* **The loop is cut to whole bars.** An arbitrary upload does not end where a
  bar ends, so looping it puts a seam in the middle of a beat and the whole
  thing limps. Cutting from its first beat to the last complete bar before the
  end costs a second of audio and buys a loop point that lands where a listener
  expects one.

* **Transposition follows the relative key, not the tonic.** A minor loop under
  a major song does not want to be moved to that major tonic — it wants the
  relative minor of it, which shares every note. Moving the tonic instead is a
  minor third out and sounds exactly that wrong.

* **Tempo is matched to the nearest octave.** A 140 BPM beat under a 70 BPM
  song is already in time; asking it to halve would destroy it. Folding the
  ratio by twos until it is nearest 1 bounds every stretch to between 0.71x and
  1.41x, which is the range WSOLA still sounds like music in.

* **Pitch and tempo are separated.** `asetrate` moves both together (it is a
  tape speed change), so the tempo filter afterwards has to pay back exactly
  what the pitch shift took — `atempo = ratio / pitch`. Get the sign of that
  wrong and everything is in tune and in the wrong tempo, or the reverse.

The plan half is pure arithmetic and unit tested on its own; the ffmpeg half is
run for real in CI, like `mixing.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .analysis import KEY_MIN_MARGIN, Track
from .audio_utils import AudioError

# There is one ffmpeg runner in this codebase and it lives in `mixing`. Importing
# it rather than writing a second one keeps the temp-file handling, the error
# text and the "map [out]" convention in a single place.
from .mixing import MixError, _ffmpeg

# 4/4, and stated as a constant rather than assumed silently. Everything this
# module is for — hip-hop, pop, EDM — is in it, and a loop cut to whole bars
# under the assumption is still cut to whole *beats* if the assumption is
# wrong, which is a much smaller error than not cutting at all.
BEATS_PER_BAR = 4
# A loop shorter than this is not a loop, it is a sample. Two bars at 120 BPM is
# four seconds.
MIN_LOOP_BARS = 2

# `atempo` takes 0.5 to 2.0 per instance in every ffmpeg this runs on. Octave
# folding already bounds the ratio to [0.71, 1.41] and the pitch division can
# push it to the edges, so this is the guard rather than the working range.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

# How far the pitch may be moved. Beyond six semitones the shorter way round is
# the other direction, so this is not a limit so much as arithmetic.
MAX_TRANSPOSE = 6

# Fade at the very end of the finished bed, so a loop cut mid-decay does not
# stop dead.
TAIL_FADE_SEC = 1.5


class BeatError(ValueError):
    """A beat that cannot be fitted: no pulse in it, or nothing to loop."""


@dataclass(frozen=True)
class Fit:
    """What has to be done to a beat before it can sit under a track.

    `reasons` is why, in words, for the container log and for the person asking
    why their beat came back a semitone away from where they left it.
    """

    semitones: int
    tempo_ratio: float
    loop_start_sec: float
    loop_length_sec: float
    reasons: tuple[str, ...] = ()

    @property
    def pitch_ratio(self) -> float:
        return 2.0 ** (self.semitones / 12.0)

    def __str__(self) -> str:
        return (
            f"{self.semitones:+d} semitone(s), tempo x{self.tempo_ratio:.3f}, "
            f"loop {self.loop_start_sec:.2f}-"
            f"{self.loop_start_sec + self.loop_length_sec:.2f}s"
            + (f" ({'; '.join(self.reasons)})" if self.reasons else "")
        )


def fold_tempo(ratio: float) -> float:
    """`ratio` halved or doubled until it is as near 1 as it can get.

    A beat at twice the tempo of the song is playing the same pulse twice as
    often, which is a thing music does on purpose. Treating that as a 2x stretch
    would be destroying a beat to fix a problem it does not have.
    """
    if ratio <= 0 or not math.isfinite(ratio):
        return 1.0
    while ratio > math.sqrt(2.0):
        ratio /= 2.0
    while ratio < 1.0 / math.sqrt(2.0):
        ratio *= 2.0
    return ratio


def transpose_to(source: Track, target: Track) -> tuple[int, str]:
    """Semitones from the beat's key to the song's, and why.

    Same mode: straight to the tonic. Different modes: to the *relative* key,
    because a key and its relative share all seven notes while a major and a
    minor a semitone apart share almost none. A minor loop under a C major song
    belongs at A minor, not at C minor.

    Zero, with a reason, whenever either key estimate is a guess —
    `Track.key_margin` exists for exactly this decision, and a transposition
    made on a coin flip is worse than none.
    """
    if source.key_margin < KEY_MIN_MARGIN:
        return 0, "beat has no clear key, left where it is"
    if target.key_margin < KEY_MIN_MARGIN:
        return 0, "song has no clear key, beat left where it is"

    destination = target.key
    note = ""
    if source.minor != target.minor:
        # The relative of the target: up a minor third from a major tonic to
        # reach its relative minor, down one to go the other way.
        destination = (target.key + 9) % 12 if source.minor else (target.key + 3) % 12
        note = f"matched to the relative {'minor' if source.minor else 'major'}"

    shift = (destination - source.key) % 12
    if shift > MAX_TRANSPOSE:
        shift -= 12
    return shift, note


def plan_fit(source: Track, target: Track) -> Fit:
    """Everything `fit` is about to do, as numbers, before any audio moves.

    Pure, so the awkward cases — a beat with no pulse, a song in no particular
    key, a loop too short to cut into bars — are decided in a unit test rather
    than in a container.
    """
    reasons: list[str] = []
    if source.bpm <= 0:
        raise BeatError("no pulse found in the beat: it cannot be fitted to anything")

    ratio = 1.0
    if target.bpm <= 0:
        reasons.append("song has no clear pulse, beat kept at its own tempo")
    else:
        ratio = fold_tempo(target.bpm / source.bpm)
        if abs(math.log2(target.bpm / source.bpm)) > 0.5:
            reasons.append(
                f"{source.bpm:.0f} against {target.bpm:.0f} BPM, matched an octave apart"
            )

    semitones, note = transpose_to(source, target)
    if note:
        reasons.append(note)

    # The loop: from the beat's own first beat to the last complete bar.
    period = 60.0 / source.bpm
    bar = period * BEATS_PER_BAR
    usable = source.duration_sec - source.beat_offset_sec
    bars = int(usable // bar)
    if bars < MIN_LOOP_BARS:
        # Too short to cut into bars — use what there is rather than refuse. A
        # one-bar loop is still a loop, and a beat that is one long phrase is
        # better looped whole than not used.
        reasons.append(f"only {usable:.1f}s of beat, looped whole")
        loop_start, loop_length = source.beat_offset_sec, usable
    else:
        loop_start, loop_length = source.beat_offset_sec, bars * bar

    if loop_length <= 0:
        raise BeatError("nothing left of the beat once it was cut to its pulse")

    return Fit(
        semitones=semitones,
        tempo_ratio=ratio,
        loop_start_sec=loop_start,
        loop_length_sec=loop_length,
        reasons=tuple(reasons),
    )


def _atempo(ratio: float) -> str:
    """One or two `atempo` stages, so the ratio is always inside ffmpeg's range."""
    if ATEMPO_MIN <= ratio <= ATEMPO_MAX:
        return f"atempo={ratio:.6f}"
    half = math.sqrt(ratio)
    return f"atempo={half:.6f},atempo={half:.6f}"


def stretch(beat_wav: bytes, fit: Fit, sample_rate: int) -> bytes:
    """The loop, cut and moved to the song's key and tempo. One ffmpeg pass.

    `asetrate` is a tape speed change: it multiplies pitch and tempo by the same
    number. So the pitch shift is done with it and the tempo it dragged along is
    handed straight back to `atempo`, which is why the ratio there is
    `tempo / pitch` and not `tempo`.

    `aresample` after `asetrate` is not optional — `asetrate` only relabels the
    stream's rate, and without a resample back everything downstream is playing
    at a rate it does not expect.
    """
    pitch = fit.pitch_ratio
    graph = (
        f"[0:a]atrim=start={fit.loop_start_sec:.6f}:"
        f"duration={fit.loop_length_sec:.6f},asetpts=PTS-STARTPTS,"
        f"asetrate={int(round(sample_rate * pitch))},aresample={sample_rate},"
        f"{_atempo(fit.tempo_ratio / pitch)}[out]"
    )
    return _ffmpeg([beat_wav], graph, ["-c:a", "pcm_s16le"], ".wav")


def lay_under(loop_wav: bytes, duration_sec: float, offset_sec: float = 0.0) -> bytes:
    """The loop repeated to `duration_sec`, starting at `offset_sec`.

    A second pass rather than one graph, because looping needs the length of
    what came *out* of the first one and ffmpeg cannot be asked mid-graph. Two
    passes of a 16-bit wav is a generation this material can afford; a guess at
    that length is not.

    `offset_sec` is where the song's first beat is, so the loop's own first beat
    — which `stretch` put at its start — lands on it.
    """
    if duration_sec <= 0:
        raise BeatError("nothing to lay a beat under")
    fade_from = max(0.0, duration_sec - TAIL_FADE_SEC)
    graph = (
        # -1 loops forever; the trim is what ends it. `size` is in samples and
        # 2^31 is "all of it" — the loop is seconds long, not hours.
        f"[0:a]aloop=loop=-1:size=2147483647,"
        f"adelay={int(round(offset_sec * 1000))}:all=1,"
        f"atrim=duration={duration_sec:.6f},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={fade_from:.6f}:d={min(TAIL_FADE_SEC, duration_sec):.6f}[out]"
    )
    return _ffmpeg([loop_wav], graph, ["-c:a", "pcm_s16le"], ".wav")


def fit(
    beat_wav: bytes,
    source: Track,
    target: Track,
    duration_sec: float,
    sample_rate: int = 44100,
) -> tuple[bytes, Fit]:
    """A beat, ready to be mixed under a vocal of `duration_sec`.

    Returns the audio and the plan that produced it, because the plan is what
    the log should say and what a person asking "why does my beat sound
    different" needs to be shown.
    """
    if not beat_wav:
        raise BeatError("no beat audio")
    plan = plan_fit(source, target)
    try:
        loop = stretch(beat_wav, plan, sample_rate)
        return lay_under(loop, duration_sec, target.beat_offset_sec), plan
    except MixError as exc:
        raise BeatError(f"could not fit the beat: {exc}") from exc


def analyse_and_fit(
    beat_wav: bytes,
    target_wav: bytes,
    duration_sec: float | None = None,
    sample_rate: int = 44100,
) -> tuple[bytes, Fit, Track, Track]:
    """`fit`, with both measurements taken here. The one call the pipeline makes.

    `duration_sec` defaults to however long the target is, which is what a
    caller replacing a backing track wants and saves it decoding the file twice
    to find out.

    Decoding happens twice — once in `analysis` at 22.05 kHz to measure, once
    in ffmpeg at full rate to process — and that is deliberate: measuring at
    half rate is four times cheaper and the answers are identical, while
    processing at half rate would throw away the top octave of somebody's beat.
    """
    from .analysis import analyse_bytes

    try:
        source = analyse_bytes(beat_wav)
        target = analyse_bytes(target_wav)
    except AudioError as exc:
        raise BeatError(f"could not read the audio to fit a beat to it: {exc}") from exc
    audio, plan = fit(
        beat_wav,
        source,
        target,
        target.duration_sec if duration_sec is None else duration_sec,
        sample_rate,
    )
    return audio, plan, source, target
