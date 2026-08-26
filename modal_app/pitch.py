"""F0 detection and the pitch shift suggestion. Ported from `chamaya00/thanh-pitch`.

Plan §7 says to reuse the YIN from the existing pitch app rather than write a
new one, so this is `detectPitch` out of that repo's `index.html`, unchanged in
algorithm and in every tuned constant: the 0.13 absolute threshold, the 0.006
RMS gate, the 0.72 clarity gate, the 0.55 fallback ceiling, the first-local-
minimum-below-threshold rule and the parabolic interpolation around it. Those
numbers were arrived at against real voices; nothing here is a fresh guess.

Two things did change, both because this runs over a whole file at once instead
of one live analyser frame at a time:

* **It is batched.** The browser version walks one 4096-sample buffer per
  animation frame with a plain loop over lags. Here the difference function is
  computed for a block of frames at once through an FFT, because a 15 minute
  track is ~18000 frames and the direct loop would be minutes of CPU.
* **The 3-frame median smoother is gone.** It existed to stop the on-screen
  needle jumping an octave. Taking the median over every voiced frame in the
  file, which is what the shift suggestion needs anyway, does the same job
  better.

Pure numpy plus ffmpeg for decoding, like `audio_utils`, so the pipeline can
call it on the CPU container and the tests can run it in CI.
"""

from __future__ import annotations

import math

import numpy as np

from .audio_utils import MODES, AudioError, check_mode, decode_audio

# 16 kHz: Nyquist is 8 kHz, well past the 1200 Hz top of the search range, and
# the difference function costs about a ninth of what it does at 48 kHz. The
# browser app ran at the AudioContext rate because that is what it was handed.
ANALYSIS_RATE = 16_000

# The original analysed 4096 samples at ~48 kHz. What matters is the duration —
# long enough to hold several periods of a low voice, short enough that the
# note does not change inside it — so the window is defined in seconds and the
# sample count follows the analysis rate.
WINDOW_SEC = 4096 / 48_000  # ~85ms
# 20 frames a second. The browser needed one per animation frame; a median over
# a whole song does not, and every frame dropped is CPU saved.
HOP_SEC = 0.05

# Ported verbatim from thanh-pitch: RANGES.speak and RANGES.sing.
F0_RANGE = {"speech": (60.0, 500.0), "singing": (60.0, 1200.0)}

# YIN_THRESHOLD in the original.
YIN_THRESHOLD = 0.13
# `if (rms < 0.006) return {freq: null}` — silence and room tone.
RMS_FLOOR = 0.006
# `if (freq && clarity > 0.72)`, applied at the call site there. clarity is
# 1 - cmnd at the chosen lag, so this is the same as cmnd < 0.28.
MIN_CLARITY = 0.72
# `if (cm[best] > 0.55) return null` — the no-dip-below-threshold fallback.
FALLBACK_CEILING = 0.55

# Frames per FFT batch. 2000 x 4096 float64 is ~65 MB, which keeps the whole
# analysis inside the CPU container's memory whatever the file length is.
BLOCK_FRAMES = 2000

# Plan §7 clamps the suggestion to one octave either way before the per-mode
# limit in `audio_utils.clamp_semitone_shift` narrows it further.
MAX_SUGGESTION = 12


def _frame(audio: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Overlapping frames as a (n_frames, window) view. Empty if too short."""
    if len(audio) < window:
        return np.zeros((0, window), dtype=np.float64)
    n_frames = 1 + (len(audio) - window) // hop
    # Strides come from the contiguous copy, not the argument: a sliced or
    # non-contiguous input would otherwise be read with the wrong step.
    contiguous = np.ascontiguousarray(audio, dtype=np.float64)
    step = contiguous.strides[0]
    return np.lib.stride_tricks.as_strided(
        contiguous, shape=(n_frames, window), strides=(step * hop, step)
    )


def _difference(frames: np.ndarray, tau_max: int) -> np.ndarray:
    """YIN's difference function d(tau) for a block of frames, via FFT.

    The browser version is a double loop:

        for tau: for i in 0..N-1: sum += (x[i] - x[i+tau])**2

    Expanded, that is `sum x[i]^2 + sum x[i+tau]^2 - 2 * sum x[i]x[i+tau]`: two
    prefix-sum lookups and one correlation, and the correlation is what the FFT
    is for. Same values, same fixed window `N = W - tau_max` for every lag.
    """
    n_frames, window = frames.shape
    length = window - tau_max

    power = np.zeros((n_frames, window + 1))
    np.cumsum(frames**2, axis=1, out=power[:, 1:])
    taus = np.arange(tau_max + 1)
    energy = power[:, length : length + 1] + (power[:, taus + length] - power[:, taus])

    # `length + window - 1` is the furthest index the correlation touches, so
    # anything at or above it leaves the circular wrap outside the lags we read.
    nfft = 1 << math.ceil(math.log2(length + window))
    spectrum = np.conj(np.fft.rfft(frames[:, :length], n=nfft)) * np.fft.rfft(frames, n=nfft)
    correlation = np.fft.irfft(spectrum, n=nfft)[:, : tau_max + 1]

    return np.maximum(energy - 2.0 * correlation, 0.0)


def _cumulative_mean(difference: np.ndarray, tau_min: int) -> np.ndarray:
    """d'(tau), normalised the way the original does it — from `tau_min`, not 1."""
    window = difference[:, tau_min:]
    running = np.cumsum(window, axis=1)
    counts = np.arange(1, window.shape[1] + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(running > 0, window * counts / running, 1.0)
    return normalised


def _pick(row: np.ndarray) -> int:
    """The chosen lag within a d' row, or -1 when the frame is unvoiced.

    Straight from the original: the first local minimum below the threshold,
    walking down while d' keeps falling; failing that the global minimum, but
    only if it is convincing enough.
    """
    below = np.flatnonzero(row < YIN_THRESHOLD)
    if below.size:
        index = int(below[0])
        while index + 1 < row.size and row[index + 1] < row[index]:
            index += 1
        return index
    best = int(np.argmin(row))
    return -1 if row[best] > FALLBACK_CEILING else best


def _interpolate(row: np.ndarray, index: int) -> float:
    """Parabolic interpolation around the minimum, in lag units."""
    if index <= 0 or index >= row.size - 1:
        return float(index)
    before, middle, after = row[index - 1], row[index], row[index + 1]
    denominator = 2.0 * (2.0 * middle - after - before)
    return float(index) if denominator == 0 else index + (after - before) / denominator


def f0_voiced(audio: np.ndarray, sample_rate: int, mode: str = "singing") -> np.ndarray:
    """F0 in Hz for every voiced frame. Unvoiced frames are simply not returned.

    Dropping them rather than returning zeros is the point of the whole
    function: a median taken over silence is the bug plan §7 warns about, and
    the caller cannot make that mistake with values that were never there.
    """
    f0_min, f0_max = F0_RANGE[check_mode(mode)]
    window = int(round(WINDOW_SEC * sample_rate))
    hop = max(1, int(round(HOP_SEC * sample_rate)))

    tau_min = max(2, int(sample_rate // f0_max))
    tau_max = min(window // 2, math.ceil(sample_rate / f0_min))
    if tau_max <= tau_min + 2:
        return np.zeros(0)

    frames = _frame(np.asarray(audio, dtype=np.float64), window, hop)
    voiced: list[np.ndarray] = []

    for start in range(0, len(frames), BLOCK_FRAMES):
        block = frames[start : start + BLOCK_FRAMES]
        # DC removal then the RMS gate, exactly as the original opens.
        block = block - block.mean(axis=1, keepdims=True)
        loud = np.sqrt((block**2).mean(axis=1)) >= RMS_FLOOR
        if not loud.any():
            continue
        block = block[loud]

        normalised = _cumulative_mean(_difference(block, tau_max), tau_min)
        for row in normalised:
            index = _pick(row)
            if index < 0 or 1.0 - row[index] <= MIN_CLARITY:
                continue
            lag = _interpolate(row, index) + tau_min
            frequency = sample_rate / lag if lag > 0 else 0.0
            if f0_min <= frequency <= f0_max:
                voiced.append(frequency)

    return np.asarray(voiced, dtype=np.float64)


def median_f0(audio: np.ndarray, sample_rate: int, mode: str = "singing") -> float | None:
    """Median F0 over voiced frames only, or None when nothing was voiced."""
    voiced = f0_voiced(audio, sample_rate, mode)
    return float(np.median(voiced)) if voiced.size else None


def median_f0_bytes(data: bytes, mode: str = "singing") -> float | None:
    """`median_f0` for an encoded file, decoded through ffmpeg."""
    return median_f0(decode_audio(data, ANALYSIS_RATE), ANALYSIS_RATE, mode)


def semitones_between(f0_source: float | None, f0_reference: float | None) -> int:
    """Semitones from source to reference, clamped to an octave. 0 if unknown."""
    if not f0_source or not f0_reference or f0_source <= 0 or f0_reference <= 0:
        return 0
    shift = round(12.0 * math.log2(f0_reference / f0_source))
    return int(max(-MAX_SUGGESTION, min(MAX_SUGGESTION, shift)))


def suggest_semitone_shift(source_wav: bytes, reference_wav: bytes, mode: str = "singing") -> int:
    """How far to move the source's pitch to sit where the reference sits.

    Run this on the **whole** vocal stem before chunking (plan §7), once per
    job. A per-chunk value would drift audibly across a song, which the plan
    calls the most common bug in this kind of app.

    Both sides are measured with the same F0 range so neither is truncated
    relative to the other. Returning 0 for "could not tell" is deliberate: no
    shift is the one answer that is never actively wrong.
    """
    check_mode(mode)
    try:
        source = median_f0_bytes(source_wav, mode)
        reference = median_f0_bytes(reference_wav, mode)
    except AudioError:
        # A suggestion is a convenience. If the audio will not decode, the
        # conversion step is about to say so with a much better message.
        return 0
    return semitones_between(source, reference)


__all__ = [
    "ANALYSIS_RATE",
    "F0_RANGE",
    "MODES",
    "f0_voiced",
    "median_f0",
    "median_f0_bytes",
    "semitones_between",
    "suggest_semitone_shift",
]
