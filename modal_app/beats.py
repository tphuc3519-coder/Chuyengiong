"""Making one piece of music sit under another one.

    beat ──► plan_fit(beat, bài) ──► Fit(dịch bao nhiêu, kéo bao nhiêu)
         └─► fit() ──► cắt tròn ô nhịp ──► dịch tone ──► kéo tempo
                   ──► lặp cho đủ dài ──► đẩy vào đúng phách ──► wav

`analysis.py` measures; this module acts on the measurements. Between them they
are the whole answer to "đổi beat" — and the reason the answer is two modules is
that measuring is arithmetic anybody can check, while acting on it is four
ffmpeg filters whose order matters.

Five things happen, and each is one decision:

* **The loop is cut to whole bars, from a bar line.** An arbitrary upload does
  not end where a bar ends, so looping it puts a seam in the middle of a beat
  and the whole thing limps. Cutting from its first *downbeat* to the last
  complete bar before the end costs a second of audio and buys a loop point
  that lands where a listener expects one.

* **The bar lines are matched, not just the beats.** `analysis.downbeat` finds
  which of four beats carries the bar line on each side, and the loop's own bar
  one is placed on the song's. Aligning to the nearest *beat* instead — which
  is what this module did first — is in time and in the wrong place: the bed's
  kick lands on the song's beat two and stays there for three minutes. Where
  either side has no clear bar line the alignment falls back to the beat, and
  `Fit.reasons` says so.

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

import io
import math
import wave
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

# A fade this long on each end of the loop, so the splice never clicks.
#
# Six milliseconds is a quarter of a cycle at 40 Hz and inaudible as a fade;
# what it removes is the step discontinuity where the end of the loop meets its
# own beginning. Cutting on a bar line puts the seam somewhere musically right
# and does nothing at all about the waveform being at +0.3 on one side of it
# and -0.4 on the other, which is a click, and a click that repeats every four
# bars is the most recognisable sound a badly looped bed makes.
#
# Deliberately not a crossfade. A crossfade would overlap the two ends and
# shorten the loop, and the loop's length is the one number in this module that
# everything else is derived from — `lay_under` reads it back off the file to
# place the bar lines.
SEAM_FADE_SEC = 0.006


class BeatError(ValueError):
    """A beat that cannot be fitted: no pulse in it, or nothing to loop."""


@dataclass(frozen=True)
class Fit:
    """What has to be done to a beat before it can sit under a track.

    `align_sec` is where in the *song* the loop's first bar has to land — the
    song's own bar line when one was found, and its first beat when one was
    not. It is a separate field from anything about the loop because it is a
    fact about the other side of the fit, and `lay_under` is the only thing
    that reads it.

    `reasons` is why, in words, for the container log and for the person asking
    why their beat came back a semitone away from where they left it.
    """

    semitones: int
    tempo_ratio: float
    loop_start_sec: float
    loop_length_sec: float
    reasons: tuple[str, ...] = ()
    align_sec: float = 0.0

    @property
    def pitch_ratio(self) -> float:
        return 2.0 ** (self.semitones / 12.0)

    def __str__(self) -> str:
        return (
            f"{self.semitones:+d} semitone(s), tempo x{self.tempo_ratio:.3f}, "
            f"loop {self.loop_start_sec:.2f}-"
            f"{self.loop_start_sec + self.loop_length_sec:.2f}s "
            f"onto {self.align_sec:.2f}s"
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

    # Where each side's bar begins. `Track.bar_start_sec` is the single place
    # that decides whether the downbeat estimate was good enough to use, so
    # both of these are either a bar line or an honest fallback to a beat — and
    # the reasons below say which, because "the beat is in time but sits a beat
    # into the bar" is exactly the complaint this answers.
    loop_start = source.bar_start_sec
    align = target.bar_start_sec
    if not source.has_downbeat:
        reasons.append("no clear bar line in the beat, cut from its first beat")
    if not target.has_downbeat:
        reasons.append("no clear bar line in the song, beat placed on its first beat")

    # The loop: from that bar line to the last complete bar before the end.
    period = 60.0 / source.bpm
    bar = period * BEATS_PER_BAR
    usable = source.duration_sec - loop_start
    bars = int(usable // bar)
    if bars < MIN_LOOP_BARS:
        # Too short to cut into bars — use what there is rather than refuse. A
        # one-bar loop is still a loop, and a beat that is one long phrase is
        # better looped whole than not used.
        reasons.append(f"only {usable:.1f}s of beat, looped whole")
        loop_length = usable
    else:
        loop_length = bars * bar

    if loop_length <= 0:
        raise BeatError("nothing left of the beat once it was cut to its pulse")

    return Fit(
        semitones=semitones,
        tempo_ratio=ratio,
        loop_start_sec=loop_start,
        loop_length_sec=loop_length,
        reasons=tuple(reasons),
        align_sec=align,
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

    The seam fades are **not** applied here, and that is a decision rather than
    an omission: `atempo` does not return exactly the length the arithmetic
    says it will — at a ratio of 1.0 a two second loop came back 1.99893 s
    long — so a fade-out scheduled from a computed length ends after the audio
    does and gets cut off part way down, which is the click it was added to
    remove. `lay_under` measures the file and fades it there.
    """
    pitch = fit.pitch_ratio
    graph = (
        f"[0:a]atrim=start={fit.loop_start_sec:.6f}:"
        f"duration={fit.loop_length_sec:.6f},asetpts=PTS-STARTPTS,"
        f"asetrate={int(round(sample_rate * pitch))},aresample={sample_rate},"
        f"{_atempo(fit.tempo_ratio / pitch)}[out]"
    )
    return _ffmpeg([beat_wav], graph, ["-c:a", "pcm_s16le"], ".wav")


def _wav_seconds(wav: bytes) -> float:
    """How long a wav this module just wrote is.

    `wave` from the standard library, the same way `mixing._sample_rate` reads
    a rate: the only file this is ever handed is the plain PCM one `stretch`
    produced, so there is nothing to decode and nothing to guess.
    """
    try:
        with wave.open(io.BytesIO(wav), "rb") as src:
            rate = src.getframerate()
            return src.getnframes() / float(rate) if rate else 0.0
    except (wave.Error, EOFError) as exc:
        raise BeatError(f"could not measure the fitted loop: {exc}") from exc


def lay_under(loop_wav: bytes, duration_sec: float, align_sec: float = 0.0) -> bytes:
    """The loop repeated to `duration_sec`, with its bar one on `align_sec`.

    A second pass rather than one graph, because this needs the length of what
    came *out* of the first one and ffmpeg cannot be asked mid-graph. Two
    passes of a 16-bit wav is a generation this material can afford; a guess at
    that length is not — so the length is read back off the file rather than
    recomputed from the plan.

    **The bed starts at zero and the bar line still lands where it should**,
    and getting both at once is the whole content of this function. The obvious
    way — delay the loop until `align_sec` — puts the bar line in the right
    place and leaves the first bar or two of the song with no backing track at
    all, which is audible as the beat "coming in late" on every single job,
    because a song's first downbeat is essentially never at t=0.

    So the loop is entered *part way through* instead. Reading from
    `L - (align_sec mod L)` into an endlessly repeating loop puts bar one at
    `align_sec`, at `align_sec + L`, and so on, while the output is continuous
    from the first sample: whatever part of the bar was playing just before the
    song's first downbeat is what the song opens on, which is what a bed does.
    """
    if duration_sec <= 0:
        raise BeatError("nothing to lay a beat under")
    length = _wav_seconds(loop_wav)
    if length <= 0:
        raise BeatError("the fitted loop is empty")
    # Where to start reading, so that bar one lands on `align_sec`. The second
    # modulo turns an exact multiple back into 0 rather than into `length`,
    # which `atrim` would read as "start after the end".
    start = (length - (max(0.0, align_sec) % length)) % length

    # The seam fades, applied to the loop **before** it is repeated, so every
    # copy carries them and every splice between two copies is a ramp to zero
    # and back rather than a step. This is the only place they can be applied
    # from a length that is known rather than computed — see `stretch`.
    seam = min(SEAM_FADE_SEC, length / 8.0)
    fade_from = max(0.0, duration_sec - TAIL_FADE_SEC)
    graph = (
        f"[0:a]afade=t=in:st=0:d={seam:.6f},"
        f"afade=t=out:st={length - seam:.6f}:d={seam:.6f},"
        # -1 loops forever; the trim is what ends it. `size` is in samples and
        # 2^31 is "all of it" — the loop is seconds long, not hours.
        f"aloop=loop=-1:size=2147483647,"
        f"atrim=start={start:.6f}:duration={duration_sec:.6f},asetpts=PTS-STARTPTS,"
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
        return lay_under(loop, duration_sec, plan.align_sec), plan
    except MixError as exc:
        raise BeatError(f"could not fit the beat: {exc}") from exc


# --- tonal balance --------------------------------------------------------

# Where the shelf sits, and what it is aiming for.
#
# **These numbers came out of one measurement of one finished job**, and it is
# worth writing down what it said because nothing else in this module needed a
# number like this:
#
#     40-120 Hz  71.9 %      (a human rock arrangement of the same song: 23.5 %)
#     2-6 kHz     5.3 %      (the same reference: 16.0 %)
#
# That is not a mix with too much bass. That is a mix that is almost *only*
# bass — one sustained low note carrying three quarters of the energy, with the
# band a voice lives in reduced to a fifth of where it should be. It reads as
# mud with the singer somewhere behind it.
#
# Where it comes from is the honest part: `BeatGenerator.generate` used to
# normalise its output by **peak**, and peak normalisation of a bass-heavy
# signal sets the whole level by the bass and leaves everything above it small.
# The generator's output is unmastered by definition — a diffusion model has no
# reason to hand back a balanced mix — so something has to condition it, and
# nothing did.
#
# `LOW_SHARE_TARGET` is a starting point and not a measurement. It needs an ear
# on a real job, exactly like `beatgen`'s two noise levels do.
SUB_HZ = 30.0
LOW_HZ = 120.0
LOW_SHARE_TARGET = 0.30
BALANCE_RMS = 0.14
BALANCE_PEAK = 0.89


def balance(beat_wav: bytes, sample_rate: int = 44100) -> tuple[bytes, str]:
    """Take the mud out of a generated bed. Returns the audio and what it did.

    **Runs after `fit`, not before, and that ordering is the whole reason this
    is a separate function.** `stretch` transposes with `asetrate`, which moves
    every frequency at once — a bed measured before a five semitone drop is a
    bed measured at the wrong place. Whatever this reads is what the mix will
    actually get.

    Three steps, in order:

    1. Everything below `SUB_HZ` goes. A backing bed has nothing to say down
       there and a voice has nothing to compete with it; what lives there is
       rumble, and rumble that survives into `mixing` is rumble that
       `loudnorm` then turns *up*, because K-weighting barely hears it.
    2. If more than `LOW_SHARE_TARGET` of the power sits below `LOW_HZ`, the
       low band is pulled back until that much does. A shelf rather than a
       filter with a corner: cosine crossovers at both ends, because a brick
       wall in the frequency domain is a long impulse response in the time one.
    3. Level set by **RMS**, with the peak only as a ceiling. This is the step
       that replaces peak normalisation, and it is the one that stops a single
       low note from deciding how loud the whole bed is.

    Only generated beds go through this. An uploaded beat is somebody's
    finished production and re-balancing it would be this module deciding it
    knows better than whoever mixed it.

    **Stereo in, stereo out, and one shelf for both channels.** The share is
    measured on the two channels summed and the same gain curve is applied to
    each, which is the only version of this that does not move the image: a
    shelf fitted per channel would pull back whichever side happened to carry
    more bass and swing the arrangement towards the other one. Level is set
    from the peak and RMS *across* both channels for the same reason.
    """
    import numpy as np

    from .audio_utils import decode_wav_channels, encode_wav_channels, to_pcm_wav

    try:
        decoded, _ = decode_wav_channels(to_pcm_wav(beat_wav, sample_rate))
        audio = np.asarray(decoded, dtype=np.float64)
    except AudioError as exc:
        # Same wrapping `analyse_and_fit` does: everything a caller of this
        # module has to catch is a `BeatError`.
        raise BeatError(f"could not read the beat to balance it: {exc}") from exc
    if not len(audio):
        raise BeatError("no beat audio to balance")

    frames = len(audio)
    freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    # One measurement for the whole bed, taken on the channels summed.
    power = np.abs(np.fft.rfft(audio.sum(axis=1))) ** 2
    total = float(power.sum())
    if total <= 0:
        raise BeatError("the generated beat is silent")

    share = float(power[freqs < LOW_HZ].sum() / total)
    gain = np.ones_like(freqs)
    # The sub goes to nothing, over a ramp rather than a cliff.
    floor = SUB_HZ * 0.6
    gain[freqs < floor] = 0.0
    ramp = (freqs >= floor) & (freqs < SUB_HZ)
    gain[ramp] = 0.5 - 0.5 * np.cos(np.pi * (freqs[ramp] - floor) / (SUB_HZ - floor))

    shelf = 1.0
    if share > LOW_SHARE_TARGET:
        # The amplitude factor that leaves `LOW_SHARE_TARGET` of the power
        # below `LOW_HZ`, holding everything above it exactly where it is.
        rest = 1.0 - share
        shelf = float(np.sqrt((LOW_SHARE_TARGET / (1.0 - LOW_SHARE_TARGET)) * rest / share))
        gain[(freqs >= SUB_HZ) & (freqs < LOW_HZ)] *= shelf
        gain[freqs < SUB_HZ] *= shelf
        # An octave to come back up over, so the shelf has no edge on it.
        knee = (freqs >= LOW_HZ) & (freqs < LOW_HZ * 2)
        blend = 0.5 - 0.5 * np.cos(np.pi * (freqs[knee] - LOW_HZ) / LOW_HZ)
        gain[knee] = shelf + (1.0 - shelf) * blend

    # One channel at a time, so the biggest array alive at once is one
    # channel's spectrum rather than the whole bed's. Four minutes of stereo at
    # 44.1 kHz is 10.6 million frames a side, and a complex128 spectrum of that
    # is 170 MB — worth not holding two of.
    out = np.empty_like(audio)
    for channel in range(audio.shape[1]):
        out[:, channel] = np.fft.irfft(np.fft.rfft(audio[:, channel]) * gain, n=frames)

    rms = float(np.sqrt((out**2).mean()))
    if rms > 0:
        out = out * (BALANCE_RMS / rms)
    peak = float(np.abs(out).max())
    if peak > BALANCE_PEAK:
        out = out * (BALANCE_PEAK / peak)

    note = (
        f"low {share * 100:.0f}% -> {LOW_SHARE_TARGET * 100:.0f}% "
        f"(shelf {20 * math.log10(max(shelf, 1e-6)):+.1f} dB), rms {BALANCE_RMS:.2f}, "
        f"{audio.shape[1]}ch"
    )
    return encode_wav_channels(np.asarray(out, dtype=np.float32), sample_rate), note


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
