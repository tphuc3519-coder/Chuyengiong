"""How fast a track is and what key it is in.

    audio ──► onset_envelope ──► tempo()  ──► (bpm, chỗ phách đầu)
          └─► chroma ─────────► key()    ──► (chủ âm, trưởng/thứ)

The two measurements a beat has to match before it can sit under somebody's
voice. `pitch.py` already answers "what note is this person on"; this answers
"what is the song doing", which is a different question with different tools —
F0 is about one voice at a time, and both of these are about everything at once.

numpy plus nothing, like `pitch.py` and `audio_utils.py`: it runs on the small
CPU image inside the pipeline, and every rule in it is checked in CI against
signals whose answer is arithmetic — a click track at exactly 100 BPM, a chord
progression that is unambiguously in one key.

**Both are estimates and both have a characteristic way of being wrong**, which
is written down here rather than discovered later:

* **Tempo comes in octaves.** 140 BPM and 70 BPM are the same clicks counted
  differently, and no amount of signal processing settles which one a listener
  would tap. A log-normal prior around 120 BPM is the usual tie-breaker and it
  is what `TEMPO_PRIOR_BPM` is; `beats.py` then treats a factor of two as free,
  because a beat at half the tempo of a song is not a beat that needs
  stretching.

* **Key detection reads the loudest notes, not the harmony.** A track whose
  chorus sits on the relative minor will get scored towards that, and a
  Krumhansl-Schmuckler correlation cannot tell a key from its relative — they
  share all seven notes. That is why `key()` returns the correlation margin as
  well as the answer: a small margin means "do not transpose anything on the
  strength of this".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio_utils import AudioError, decode_audio

# 22.05 kHz: everything either measurement cares about is under 5 kHz, and the
# STFT costs a quarter of what it does at 44.1.
ANALYSIS_RATE = 22_050
FRAME = 2048
# ~23 ms per frame, 43 frames a second. A 200 BPM beat is 3.3 frames apart,
# which is the resolution floor and the reason `BPM_MAX` is where it is.
HOP = 512

# --- tempo ----------------------------------------------------------------

BPM_MIN = 60.0
BPM_MAX = 200.0
# Where a listener's tap lands when the signal is ambiguous. 120 is the
# conventional centre and the spread is wide — this breaks ties, it does not
# overrule evidence.
TEMPO_PRIOR_BPM = 120.0
TEMPO_PRIOR_OCTAVES = 1.0

# A tempo hypothesis is scored at its multiples as well as at itself.
#
# This is the fix for the error that actually damages a backing track, and it
# is worth naming precisely because it is *not* the octave error everybody
# expects. On a real rock arrangement the autocorrelation peaks almost equally
# at the beat period and at **one and a half times** it, because a backbeat
# puts strong onsets on both grids. Measured on a 154 BPM song, plain
# autocorrelation scored 103.4 BPM at 0.4817 and 152.0 at 0.4774 — it picked
# the wrong one by 0.9%.
#
# A 3:2 error is the one that cannot be lived with. Two bars of a bed at 103
# span three bars of a song at 154: it does not drift, it is simply in a
# different metre. An octave error is benign by comparison — a bed at half
# tempo lands on every other beat, which is what half-time is — and
# `beats.fold_tempo` absorbs it anyway.
#
# Multiples separate the two cases exactly. A true period has support at 1x,
# 2x, 3x and 4x of itself; a lag 1.5x too long shares only its even multiples,
# so its odd ones collapse. On the same song: 152 scores 0.506/0.485/0.378/0.504
# across the four, and 103 scores 0.493/0.378/0.252/0.505 — the difference is
# not marginal once all four are counted. It leaves the octave ambiguity
# untouched, which is correct: every multiple of 2P is also a multiple of P, so
# no amount of comb scoring can tell them apart, and nothing should pretend to.
TEMPO_HARMONICS = (1, 2, 3, 4)
# Onset envelope smoothing window, in frames: the local mean that gets
# subtracted before autocorrelation. ~0.35s, long enough to span a beat and
# short enough to follow a build.
ONSET_MEAN_FRAMES = 15

# How far ahead of the sound the onset envelope fires, in frames.
#
# Not a fudge factor — it falls out of what spectral flux is. Entry `i` of the
# envelope compares frame `i+1` against frame `i`, and the only samples in one
# and not the other are `[i*HOP + FRAME, (i+1)*HOP + FRAME)`. So the sound that
# made entry `i` rise happened around sample `i*HOP + FRAME`, which is `FRAME /
# HOP` frames later than entry `i`'s own position.
#
# It does not matter for the tempo — every entry is early by the same amount,
# so the spacing between them is untouched — and it matters completely for the
# phase, which is the one number a beat gets aligned by. Uncorrected it puts
# every downbeat 70 ms early, which is not subtle when a beat is playing under
# somebody singing.
ONSET_LEAD_FRAMES = FRAME / HOP

# How far the onset envelope has to rise above the spectrum it came from before
# the track counts as having a pulse at all.
#
# Without a gate, `tempo()` answers every time it is asked. A held organ chord
# comes back at 108 BPM — the flux is floating-point dust, the autocorrelation
# of dust is as periodic as anything else, and nothing downstream can tell that
# answer from a real one. A beat then gets stretched to a tempo that was never
# in the recording.
#
# The envelope is divided by the mean level of the spectrum it was computed
# from, so this is a ratio and not a level: it does not move when the recording
# gets louder. Measured, a click track sits near 6000 and a chord progression
# near 3000, while a steady tone is at 3 and white noise at 21. 50 is therefore
# not a tuned number, it is the middle of a gap two orders of magnitude wide,
# and it is placed low on purpose — a missed pulse costs a beat its tempo
# match, an invented one costs it its tempo.
PULSE_FLOOR = 50.0

# --- key ------------------------------------------------------------------

# Where pitch classes are worth counting. Below 65 Hz a bass note's harmonics
# say more than the note does, and above 2 kHz almost everything is harmonic
# rather than fundamental.
KEY_MIN_HZ = 65.0
KEY_MAX_HZ = 2000.0
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles: how much each scale degree is used in a
# major and a minor key, from the probe-tone experiments. Correlating a track's
# chroma against all 24 rotations of these is the standard method and it is
# standard because it works on anything tonal without being trained on it.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Under this gap between the best key and the runner-up, the answer is a guess.
# Relative major and minor share all seven notes, so this is not rare.
KEY_MIN_MARGIN = 0.04


@dataclass(frozen=True)
class Track:
    """What a piece of audio is doing, as the two numbers a beat must match.

    `beat_offset_sec` is where the first beat lands, not where the first *bar*
    does. Finding the downbeat is a harder problem and a wrong answer to it is
    worse than no answer — a beat aligned to the wrong quarter of a bar is
    audibly broken, while one aligned to a beat is merely not aligned to the
    bar.

    `key_margin` is how much the winning key beat the runner-up. Small means
    the track is not clearly in one key, or is sitting on the relative
    minor — either way, not something to transpose on.
    """

    bpm: float
    beat_offset_sec: float
    key: int
    minor: bool
    key_margin: float
    duration_sec: float

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.key % 12]}{'m' if self.minor else ''}"

    def __str__(self) -> str:
        return (
            f"{self.bpm:.1f} BPM, {self.key_name} "
            f"(margin {self.key_margin:.3f}), {self.duration_sec:.1f}s"
        )


def _frames(audio: np.ndarray) -> np.ndarray:
    """Overlapping windowed frames as `(n_frames, FRAME)`. Empty if too short."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < FRAME:
        return np.zeros((0, FRAME), dtype=np.float32)
    count = 1 + (len(audio) - FRAME) // HOP
    offsets = HOP * np.arange(count)[:, None] + np.arange(FRAME)[None, :]
    return audio[offsets] * np.hanning(FRAME).astype(np.float32)


def onset_envelope(audio: np.ndarray) -> np.ndarray:
    """How much new sound starts at each frame. The input to every tempo guess.

    Spectral flux: the sum of how much each frequency bin *rose* since the last
    frame, rectified so a bin going quiet contributes nothing. A note starting
    is energy appearing across many bins at once, which is exactly what this
    adds up, and it is why flux finds a drum hit that a plain envelope follower
    misses inside a loud bar.

    Log compression first, because a chorus is ten times the amplitude of a
    verse and only about twice the event: without it the autocorrelation is a
    measurement of where the loud part of the song is.

    Then the local mean is subtracted and the result rectified, which removes
    the slow drift that would otherwise dominate the autocorrelation at every
    lag equally.

    The result is divided by the spectrum's own mean level, which makes it a
    ratio rather than a quantity of anything. Nothing that reads it for tempo
    cares — autocorrelation and argmax are both blind to scale — and it is what
    lets `PULSE_FLOOR` be one number for quiet recordings and loud ones alike.
    """
    frames = _frames(audio)
    if not len(frames):
        return np.zeros(0, dtype=np.float32)
    spectrum = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    level = float(spectrum.mean())
    if level <= 0:
        return np.zeros(0, dtype=np.float32)
    flux = np.maximum(np.diff(spectrum, axis=0), 0.0).sum(axis=1) / level
    if not flux.size:
        return np.zeros(0, dtype=np.float32)

    window = min(ONSET_MEAN_FRAMES, len(flux))
    padded = np.pad(flux, (window // 2, window - window // 2 - 1), mode="edge")
    local_mean = np.convolve(padded, np.ones(window) / window, mode="valid")
    return np.maximum(flux - local_mean, 0.0).astype(np.float32)


def _autocorrelate(envelope: np.ndarray) -> np.ndarray:
    """Autocorrelation of the onset envelope, through an FFT."""
    centred = envelope - envelope.mean()
    size = 1 << math.ceil(math.log2(max(2, 2 * len(centred))))
    spectrum = np.fft.rfft(centred, n=size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: len(centred)]
    return correlation / max(correlation[0], 1e-9)


def _comb_score(correlation: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Each lag's autocorrelation plus its multiples', weighted by `1/k`.

    See `TEMPO_HARMONICS`. Normalised by the weights that actually landed
    inside the correlation, so a long period near the end of a short recording
    is not marked down for having fewer multiples to be supported at.

    **Each multiple is looked for in a small window rather than at one sample**,
    and without that the whole idea backfires. Lags are whole frames and a
    period is not: a true period of 21.53 frames is searched at lag 22, whose
    second multiple is 44 while the real peak sits at 43. The double-length
    candidate at lag 43 has multiples 86, 129, 172 that all stay aligned — so
    scoring at exact integer multiples systematically rewards *longer* lags, and
    the first version of this turned a 120 BPM click track into 60.

    The window is `ceil(k/2)` wide because that is exactly how far half a frame
    of rounding can have travelled by the k-th multiple.
    """
    total = np.zeros(len(lags), dtype=np.float64)
    weight = np.zeros(len(lags), dtype=np.float64)
    for harmonic in TEMPO_HARMONICS:
        tolerance = -(-harmonic // 2)  # ceil, and the reach of a rounded lag
        centre = lags * harmonic
        inside = centre + tolerance < len(correlation)
        if not inside.any():
            break
        offsets = np.arange(-tolerance, tolerance + 1)
        window = correlation[centre[inside][:, None] + offsets[None, :]]
        share = 1.0 / harmonic
        total[inside] += share * window.max(axis=1)
        weight[inside] += share
    return total / np.maximum(weight, 1e-9)


def _refine(correlation: np.ndarray, lag: int) -> float:
    """The autocorrelation peak's position to better than one frame.

    A frame is 23 ms, so at 120 BPM the true period lands between lag 21 and
    lag 22 and rounding picks 117.5 or 123 — a 2.5% error, which over three
    minutes is seconds of drift between a beat and the voice on top of it. A
    parabola through the peak and its two neighbours costs three multiplications
    and removes it; `pitch.py` does the same thing to the YIN minimum for the
    same reason.
    """
    if lag <= 0 or lag >= len(correlation) - 1:
        return float(lag)
    before, middle, after = correlation[lag - 1], correlation[lag], correlation[lag + 1]
    denominator = before - 2.0 * middle + after
    if denominator == 0:
        return float(lag)
    return float(lag) + 0.5 * (before - after) / denominator


def _fit_grid(envelope: np.ndarray, period: float, offset: float) -> tuple[float, float]:
    """Least-squares refit of the beat grid to the onsets it actually lands on.

    Autocorrelation gives a period good to a few tenths of a percent, which
    sounds like nothing and is not: 0.3% over a three minute song is half a
    second of drift between a beat and the voice on top of it, and drift is the
    one timing error a listener cannot ignore.

    The fix is to stop treating the period as one measurement. Walk the grid the
    autocorrelation proposed, find the loudest onset near each predicted beat,
    and fit a straight line through the ones that were found: the slope is the
    period and the intercept is the phase, both now estimated from the whole
    track at once instead of from a single peak in a correlation. A hundred
    beats spanning three minutes pin the tempo far more tightly than any one of
    them could.

    Falls back to what it was given whenever there is not enough to fit — a
    handful of beats, or a track with no onsets near the grid at all.
    """
    if period <= 0:
        return period, offset
    search = max(1, int(round(period / 4)))
    indices: list[float] = []
    positions: list[float] = []
    for step in range(int((len(envelope) - offset) / period) + 1):
        centre = int(round(offset + step * period))
        low, high = max(0, centre - search), min(len(envelope), centre + search + 1)
        if high <= low:
            continue
        window = envelope[low:high]
        if window.max() <= 0:
            continue
        indices.append(float(step))
        positions.append(float(low + int(window.argmax())))
    if len(positions) < 4:
        return period, offset

    design = np.vstack([np.asarray(indices), np.ones(len(indices))]).T
    slope, intercept = np.linalg.lstsq(design, np.asarray(positions), rcond=None)[0]
    # A fit that moved the period more than a quarter is not a refinement, it is
    # a different answer arrived at by locking onto the wrong onsets.
    if not 0.75 * period <= slope <= 1.25 * period:
        return period, offset
    return float(slope), float(intercept)


def tempo(audio: np.ndarray, sample_rate: int = ANALYSIS_RATE) -> tuple[float, float]:
    """`(bpm, seconds to the first beat)`.

    The period comes from the strongest autocorrelation lag inside the BPM
    range, weighted by the log-normal prior and then interpolated to
    sub-frame accuracy — see `_refine`, without which every answer is rounded
    to the nearest 23 ms and drifts. The phase then comes from asking,
    for each possible offset within one period, how much onset energy lands on
    the beats it implies — the offset that catches the most is where the beat
    is.

    Returns `(0.0, 0.0)` for audio with no discernible pulse, which the callers
    treat as "do not stretch this": silence, a held pad, a spoken word file.
    That case is decided by `PULSE_FLOOR` before any of the above runs, because
    every step after it will produce a number whether or not there was one.
    """
    envelope = onset_envelope(audio)
    frames_per_sec = sample_rate / HOP
    if len(envelope) < 4 or not envelope.any():
        return 0.0, 0.0
    # Nothing starts in this recording — a held chord, a pad, a room. Saying so
    # is the answer; see `PULSE_FLOOR`.
    if float(envelope.max()) < PULSE_FLOOR:
        return 0.0, 0.0

    correlation = _autocorrelate(envelope)
    low = max(1, int(round(frames_per_sec * 60.0 / BPM_MAX)))
    high = min(len(correlation) - 1, int(round(frames_per_sec * 60.0 / BPM_MIN)))
    if high <= low:
        return 0.0, 0.0

    lags = np.arange(low, high + 1)
    candidates = 60.0 * frames_per_sec / lags
    prior = np.exp(-0.5 * (np.log2(candidates / TEMPO_PRIOR_BPM) / TEMPO_PRIOR_OCTAVES) ** 2)
    scored = _comb_score(correlation, lags) * prior
    if not np.isfinite(scored).any() or scored.max() <= 0:
        return 0.0, 0.0

    period = _refine(correlation, int(lags[int(scored.argmax())]))
    if period <= 0:
        return 0.0, 0.0

    # Phase: of the grids this period allows, which one collects the most
    # onset energy. Sampled at rounded frame positions because the period is
    # fractional now — a grid that drifts by half a frame over a bar is still
    # the same grid.
    beats = np.arange(0, len(envelope) / period)
    energy = [
        float(envelope[np.rint(offset + beats * period).astype(int) % len(envelope)].sum())
        for offset in range(int(round(period)))
    ]
    if not energy:
        return 0.0, 0.0
    # Interpolated for the same reason the period is: the search is one frame
    # apart, and 23 ms of slop in where the beat falls is audible slop under a
    # voice. Circular, because offset and offset+period are the same grid.
    grid = np.asarray(energy)
    best = int(grid.argmax())
    refined = best + _refine(np.roll(grid, 1 - best)[:3], 1) - 1.0

    # Both numbers again, this time from every beat in the track rather than
    # from one peak in a correlation.
    period, refined = _fit_grid(envelope, period, refined)
    if period <= 0:
        return 0.0, 0.0

    # Back out the envelope's lead, and wrap: a negative time to the first beat
    # is not a thing.
    seconds = (refined + ONSET_LEAD_FRAMES) / frames_per_sec
    return float(60.0 * frames_per_sec / period), float(seconds % (period / frames_per_sec))


def chroma(audio: np.ndarray, sample_rate: int = ANALYSIS_RATE) -> np.ndarray:
    """Energy per pitch class over the whole file, summing to 1.

    Every FFT bin between `KEY_MIN_HZ` and `KEY_MAX_HZ` is assigned to the
    semitone it is nearest and its magnitude added there. Crude next to a
    constant-Q transform, and enough: a key estimate is a 12-way decision made
    over minutes of audio, so bin-to-semitone rounding error at the bottom of
    the range averages out long before it changes the answer.
    """
    frames = _frames(audio)
    if not len(frames):
        return np.zeros(12)
    magnitude = np.abs(np.fft.rfft(frames, axis=1)).sum(axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate)

    usable = (freqs >= KEY_MIN_HZ) & (freqs <= KEY_MAX_HZ)
    if not usable.any():
        return np.zeros(12)
    classes = np.rint(69 + 12 * np.log2(freqs[usable] / 440.0)).astype(int) % 12
    totals = np.bincount(classes, weights=magnitude[usable], minlength=12)
    return totals / total if (total := totals.sum()) > 0 else np.zeros(12)


def key(audio: np.ndarray, sample_rate: int = ANALYSIS_RATE) -> tuple[int, bool, float]:
    """`(tonic 0-11, is minor, margin over the runner-up)`.

    Correlation against all 24 rotations of the Krumhansl-Schmuckler profiles.
    The margin is returned because the method's known weakness is exactly the
    case where it matters: a key and its relative share every note, so the two
    correlate almost equally on a track that uses both, and a transposition
    decided by a coin flip moves a beat a minor third away from the song.
    """
    weights = chroma(audio, sample_rate)
    if not weights.any():
        return 0, False, 0.0

    scores = []
    for minor in (False, True):
        profile = MINOR_PROFILE if minor else MAJOR_PROFILE
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            scores.append((float(np.corrcoef(weights, rotated)[0, 1]), tonic, minor))
    scores = [item for item in scores if np.isfinite(item[0])]
    if not scores:
        return 0, False, 0.0

    scores.sort(reverse=True)
    best, tonic, minor = scores[0]
    return tonic, minor, float(best - scores[1][0]) if len(scores) > 1 else 0.0


def analyse(audio: np.ndarray, sample_rate: int = ANALYSIS_RATE) -> Track:
    """Both measurements plus the duration, as one record."""
    bpm, offset = tempo(audio, sample_rate)
    tonic, minor, margin = key(audio, sample_rate)
    return Track(
        bpm=bpm,
        beat_offset_sec=offset,
        key=tonic,
        minor=minor,
        key_margin=margin,
        duration_sec=len(audio) / float(sample_rate),
    )


def analyse_bytes(data: bytes) -> Track:
    """`analyse` for an encoded file, decoded through ffmpeg."""
    if not data:
        raise AudioError("nothing to analyse")
    return analyse(decode_audio(data, ANALYSIS_RATE), ANALYSIS_RATE)
