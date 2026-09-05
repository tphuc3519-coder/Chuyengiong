"""Cleaning the voice sample, before it becomes anybody's timbre.

    upload ──► clean() ──► prepare() ──► [main window, extra windows]
                                              │            └─► style embedding
                                              └─────────────► mel + content

`audio_utils.prepare_reference` decides *which* twenty seconds of a recording
the model is shown. This module decides what those twenty seconds sound like,
and it exists because of the one thing about Seed-VC that surprises everybody
who tries it with a phone recording: it does not copy a voice, it copies a
**sample**. Whatever else is in the reference — the hiss of a cheap preamp, a
laptop fan, the hum off a charger, the boom of a small room — is not background
to the model. It is part of the timbre it was asked to reproduce, so it comes
out fused into every syllable of the result, and the result then sounds
processed, grainy, artificial: the thing people describe as "nghe bị AI".

Nothing here is a model. It is three mechanical corrections, in the order they
have to happen:

* **Rumble goes first.** Below ~70 Hz there is nothing in a human voice and
  quite a lot of desk knock, footfall, door slam and mains hum. It is inaudible
  on a laptop speaker, it is a large fraction of the sample's energy, and every
  level decision downstream — the normalisation here, `loudnorm` at the end —
  is made worse by counting it.

* **The noise floor is estimated and subtracted.** Per frequency bin, the level
  the sample sits at when nobody is saying anything is a low percentile of that
  bin over time — speech is loud and intermittent, room tone is quiet and
  constant, so the percentile lands on the room. Subtracting it with a spectral
  floor (never to zero, which is what makes the "underwater" artefact
  everybody's noise reduction is remembered for) leaves the voice and takes the
  room.

* **The level is set.** Not for loudness — `loudnorm` does that at the very end
  — but because the reference and the source arrive at the model at whatever
  level two different microphones happened to produce, and a quiet reference is
  a reference the encoder sees less of.

And one thing that is not a correction: **a long sample is worth more than the
best twenty seconds of it.** The speaker embedding is an average over a window,
so a window is a sample of the speaker rather than the speaker. Where the
recording is long enough to hold more than one, `prepare` hands back the extras
and `conversion` averages the embeddings — which is what speaker-verification
enrolment has always done, and for the same reason: it cancels what is
particular to one stretch of a recording and keeps what is particular to the
person.

numpy only, no scipy, no librosa: this runs inside the GPU container but the
whole of it is unit tested in CI on a bare box, which is where the arithmetic
above has to be checked rather than described.
"""

from __future__ import annotations

import numpy as np

from .audio_utils import (
    REFERENCE_MAX_SEC,
    SPEECH_FRAME,
    prepare_reference,
    speech_flags,
    usable_reference_window,
)

# --- the analysis window --------------------------------------------------

# 1024 samples is 23 ms at 44.1 kHz and 46 ms at 22.05 kHz. Long enough for the
# noise estimate to be about a frequency rather than about a click, short
# enough that a subtraction applied to the whole frame does not smear across a
# consonant. Half-overlap with a root-Hann window on both ends: analysis times
# synthesis is a Hann window, and Hann at 50% sums to exactly one, so an
# unmodified round trip returns the input.
FFT_SIZE = 1024
HOP = FFT_SIZE // 2

# --- the noise floor ------------------------------------------------------

# Which percentile of a bin's magnitude over time counts as "the room".
#
# Not the minimum, which is one frame and therefore one accident. 15 means the
# estimate holds as long as at least a sixth of the recording is not speech,
# which is true of anything a person recorded by pressing record, saying a
# sentence and pressing stop.
NOISE_PERCENTILE = 15.0
# Subtract a bit more than the estimate. A percentile is the *middle* of the
# quiet frames rather than their top, so 1.0 leaves audible noise behind; past
# ~2 the subtraction starts eating the tails of words.
OVERSUBTRACTION = 1.8
# How far a bin may be pulled down, as a gain rather than to silence.
#
# This number is the difference between noise reduction and the artefact
# everybody knows it by. Gating a bin to zero in one frame and back to one in
# the next leaves isolated spectral peaks flickering in the silence — "musical
# noise", the underwater burble. A floor of -14 dB removes most of the room and
# leaves what remains continuous, which is what stops it being heard as an
# effect.
SPECTRAL_FLOOR = 10.0 ** (-14.0 / 20.0)
# Frames below this many are not enough to estimate anything from; the sample
# is handed back with only the level touched.
MIN_ANALYSIS_FRAMES = 16

# --- the high pass --------------------------------------------------------

# Fully passed above, fully stopped below, cosine in between. A voice's
# fundamental bottoms out around 75 Hz for a deep male speaker, so the corner
# sits under it and the stop band is where only the room lives.
HIGHPASS_PASS_HZ = 75.0
HIGHPASS_STOP_HZ = 40.0

# --- the level ------------------------------------------------------------

# RMS of the speech (not of the file — the silences are not part of the
# measurement) that the sample is scaled to. About -24 dBFS: a comfortable
# recording level with room for the peaks of a loud syllable.
TARGET_RMS = 0.063
# Never scale a sample into clipping to reach the target. A recording that
# cannot get there without it is a recording with very high crest factor, and
# the peak matters more than the average.
MAX_PEAK = 0.97
# Samples this far below the peak are silence for the purpose of measuring the
# level, and are left out of the RMS.
LEVEL_FLOOR = 0.1

# --- picking more than one window -----------------------------------------

# How many windows the style embedding may be averaged over, the main one
# included. Three 20s windows is a minute of reference, which is about as much
# as anybody records; past that the average stops moving.
STYLE_WINDOWS = 3
# An extra window has to be *mostly* voice to be worth averaging into the
# speaker embedding. Half of it is a low bar deliberately: the point is to
# exclude the tail of a recording that is all room tone, not to insist on a
# performance.
MIN_WINDOW_SPEECH = 0.5


# --- STFT round trip ------------------------------------------------------


def _window() -> np.ndarray:
    """Root of a periodic Hann. Used for analysis and synthesis both.

    Periodic (`hanning(n+1)[:-1]`) rather than symmetric: only the periodic one
    sums to a constant at 50% overlap, and the whole point of the pair is that
    a frame multiplied by it twice and overlap-added comes back unchanged.
    """
    return np.sqrt(np.hanning(FFT_SIZE + 1)[:-1]).astype(np.float32)


def _stft(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """`(spectrogram, pad)` for `audio`, zero padded at both ends by a frame.

    The padding is what lets the first and last real samples be covered by as
    many frames as the middle ones, so the overlap-add normalisation is uniform
    and there is no fade at either end.
    """
    pad = FFT_SIZE
    padded = np.pad(np.asarray(audio, dtype=np.float32), (pad, pad + FFT_SIZE))
    count = 1 + (len(padded) - FFT_SIZE) // HOP
    offsets = HOP * np.arange(count)[:, None] + np.arange(FFT_SIZE)[None, :]
    return np.fft.rfft(padded[offsets] * _window(), axis=1), pad


def _istft(spectrum: np.ndarray, length: int, pad: int) -> np.ndarray:
    """Overlap-add `spectrum` back to `length` samples.

    Divided by the summed square of the window rather than by a constant: the
    constant is only right where every frame overlaps, and the ends of the
    buffer are exactly where it is not.
    """
    window = _window()
    frames = np.fft.irfft(spectrum, n=FFT_SIZE, axis=1).astype(np.float32) * window
    total = pad + length + 2 * FFT_SIZE
    out = np.zeros(total, dtype=np.float32)
    norm = np.zeros(total, dtype=np.float32)
    square = window * window
    for index, frame in enumerate(frames):
        start = index * HOP
        out[start : start + FFT_SIZE] += frame
        norm[start : start + FFT_SIZE] += square
    out /= np.maximum(norm, 1e-8)
    return out[pad : pad + length]


def highpass_mask(sample_rate: int) -> np.ndarray:
    """Per-bin gain for the rumble filter: 0 under the stop, 1 over the pass.

    A raised cosine between the two rather than a step, because a step in the
    frequency domain is a very long impulse response in the time domain and
    rings audibly on every plosive.
    """
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)
    span = np.clip((freqs - HIGHPASS_STOP_HZ) / (HIGHPASS_PASS_HZ - HIGHPASS_STOP_HZ), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * span)).astype(np.float32)


def _smooth(gain: np.ndarray) -> np.ndarray:
    """Three-bin average along frequency.

    The other half of the musical-noise fix: neighbouring bins of a real sound
    move together, so a gain that differs sharply between two of them is the
    estimate's noise rather than the signal's. Smoothing costs a little
    reduction and buys a result that does not burble.
    """
    padded = np.pad(gain, ((0, 0), (1, 1)), mode="edge")
    return ((padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0).astype(np.float32)


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """`audio` with its own noise floor subtracted and its rumble removed.

    One STFT pass does both, because both are a per-bin gain and there is no
    reason to pay for two round trips. Too short to estimate anything from
    comes back with the rumble filter alone applied, which needs no estimate.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < FFT_SIZE * 4:
        return audio
    spectrum, pad = _stft(audio - float(audio.mean()))
    mask = highpass_mask(sample_rate)[None, :]
    if spectrum.shape[0] < MIN_ANALYSIS_FRAMES:
        return _istft(spectrum * mask, len(audio), pad)

    magnitude = np.abs(spectrum)
    noise = np.percentile(magnitude, NOISE_PERCENTILE, axis=0)[None, :]
    # The gain that would leave `magnitude - OVERSUBTRACTION * noise` behind,
    # floored so no bin is ever silenced outright.
    gain = np.clip(
        (magnitude - OVERSUBTRACTION * noise) / np.maximum(magnitude, 1e-8),
        SPECTRAL_FLOOR,
        1.0,
    ).astype(np.float32)
    return _istft(spectrum * _smooth(gain) * mask, len(audio), pad)


def normalise(audio: np.ndarray) -> np.ndarray:
    """`audio` scaled so the speech in it sits at `TARGET_RMS`, without clipping.

    Measured over the loud samples only. An RMS taken over the whole file is a
    measurement of how much of it is silence, and two recordings of the same
    voice with different amounts of pause would be scaled differently.
    """
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0:
        return audio
    loud = audio[np.abs(audio) >= peak * LEVEL_FLOOR]
    rms = float(np.sqrt(np.mean(loud.astype(np.float64) ** 2))) if loud.size else 0.0
    if rms <= 0:
        return audio
    gain = min(TARGET_RMS / rms, MAX_PEAK / peak)
    return (audio * np.float32(gain)).astype(np.float32)


def clean(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The whole correction: rumble out, room out, level set.

    Applied to the *whole* recording before a window is chosen, not to the
    window: the noise estimate wants every quiet frame it can get, and the
    twenty seconds that get used are the ones with the fewest of them.
    """
    return normalise(denoise(audio, sample_rate))


# --- choosing more than one window ----------------------------------------


def extra_ranges(
    audio: np.ndarray,
    sample_rate: int,
    taken: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    """Up to `count` further `(start, stop)` windows, none overlapping `taken`.

    Greedy and non-overlapping: the best remaining window, then the best window
    that does not touch it, and so on. A window that is less than
    `MIN_WINDOW_SPEECH` voice is not returned at all — averaging a stretch of
    room tone into a speaker embedding moves it away from the speaker, which is
    the opposite of the point.

    Ranges rather than audio, because "these windows do not overlap" is the
    property worth testing and slices cannot be asked where they came from.
    """
    limit = int(REFERENCE_MAX_SEC * sample_rate)
    if count <= 0 or len(audio) < 2 * limit:
        return []

    flags = speech_flags(audio)
    per_window = max(1, limit // SPEECH_FRAME)
    if len(flags) <= per_window:
        return []
    totals = np.convolve(flags.astype(np.int32), np.ones(per_window, dtype=np.int32), "valid")

    # Candidate start positions, in frames, minus everything the main window
    # already covers. A start is blocked when a window opening there would
    # reach into one already taken, which is `per_window` frames either side.
    free = np.ones(len(totals), dtype=bool)

    def block(start_frame: int) -> None:
        low = max(0, start_frame - per_window + 1)
        free[low : start_frame + per_window] = False

    block(taken[0] // SPEECH_FRAME)

    picked: list[tuple[int, int]] = []
    needed = int(per_window * MIN_WINDOW_SPEECH)
    for _ in range(count):
        candidates = np.where(free, totals, -1)
        best = int(candidates.argmax())
        if candidates[best] < needed:
            break
        start = best * SPEECH_FRAME
        picked.append((start, start + limit))
        block(best)
    return picked


def extra_windows(
    audio: np.ndarray,
    sample_rate: int,
    taken: tuple[int, int],
    count: int,
) -> list[np.ndarray]:
    """`extra_ranges`, sliced."""
    return [
        np.asarray(audio[start:stop], dtype=np.float32)
        for start, stop in extra_ranges(audio, sample_rate, taken, count)
    ]


def prepare(
    audio: np.ndarray, sample_rate: int, windows: int = STYLE_WINDOWS
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The reference as the converter should see it: `(main, extras)`.

    `main` is what the model is conditioned on — its mel, its content, its
    length. `extras` are only ever averaged into the speaker embedding, and are
    empty for any recording too short to hold a second window, which is most of
    them. Both have been through `clean`.

    Length validation stays in `audio_utils.prepare_reference`, which is called
    here rather than reimplemented: there is one rule about how short a
    reference may be and it should be in one place.
    """
    cleaned = clean(np.asarray(audio, dtype=np.float32), sample_rate)
    main = prepare_reference(cleaned, sample_rate)
    if windows <= 1:
        return main, []
    return main, extra_windows(
        cleaned, sample_rate, usable_reference_window(cleaned, sample_rate), windows - 1
    )
