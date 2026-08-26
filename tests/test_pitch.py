"""The ported YIN and the shift suggestion.

Synthetic tones, because a pitch detector is one of the few pieces of an audio
pipeline whose right answer is knowable in advance: a 220 Hz sawtooth-ish tone
has an F0 of 220 Hz and anything else is a bug. Real singing would only tell us
the output "sounds about right".

The two acceptance criteria from plan §7 each have a test here: a male→female
pair landing in the +10..+14 band, and a long instrumental intro not dragging
the answer down.
"""

import numpy as np
import pytest

from modal_app import pitch
from modal_app.audio_utils import AudioError

SR = pitch.ANALYSIS_RATE


def tone(freq: float, seconds: float = 2.0, harmonics: int = 4, amplitude: float = 0.5):
    """A voice-ish periodic signal: a fundamental plus a few falling harmonics."""
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, harmonics + 1))
    return (wave / np.abs(wave).max() * amplitude).astype(np.float32)


# --- detection ------------------------------------------------------------


@pytest.mark.parametrize("freq", [110.0, 147.0, 196.0, 220.0, 330.0, 440.0])
def test_a_known_tone_is_detected_within_a_tenth_of_a_percent(freq):
    detected = pitch.median_f0(tone(freq), SR, "singing")
    assert detected is not None
    assert abs(detected - freq) / freq < 0.001


def test_silence_is_not_voiced():
    """The RMS gate, ported as-is. Without it every silent frame would land in
    the median as whatever the search happened to bottom out on."""
    assert pitch.median_f0(np.zeros(SR * 2, dtype=np.float32), SR) is None


def test_quiet_noise_is_not_voiced():
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(SR * 2) * 0.001).astype(np.float32)
    assert pitch.median_f0(quiet, SR) is None


def test_loud_noise_is_not_voiced():
    """The clarity gate: noise is loud enough to pass the RMS floor and has no
    period, so d' never dips convincingly."""
    rng = np.random.default_rng(1)
    loud = (rng.standard_normal(SR * 2) * 0.3).astype(np.float32)
    assert pitch.median_f0(loud, SR) is None


def test_audio_shorter_than_one_window_is_not_voiced():
    assert pitch.median_f0(tone(220.0, 0.01), SR) is None


def test_unvoiced_frames_are_dropped_not_zeroed():
    """`f0_voiced` returns only what it found. A caller cannot average in a
    silence it never saw."""
    half = np.concatenate([np.zeros(SR, dtype=np.float32), tone(220.0, 1.0)])
    voiced = pitch.f0_voiced(half, SR, "singing")
    assert voiced.size
    assert np.all(voiced > 0)
    assert abs(float(np.median(voiced)) - 220.0) < 1.0


def test_the_range_sets_the_lag_search_so_a_high_tone_folds_down_an_octave():
    """RANGES.speak vs RANGES.sing in the original.

    The range bounds the lag search rather than filtering results, and a tone
    is periodic at every multiple of its period — so 700 Hz read with the 500 Hz
    speech ceiling comes back as its subharmonic at 350, not as nothing. Ported
    behaviour, worth pinning: it is why both sides of a comparison have to be
    measured with the same range, and why `singing` is the mode a song uses.
    """
    folded = pitch.median_f0(tone(700.0), SR, "speech")
    assert folded is not None
    assert abs(folded - 350.0) < 2.0
    assert abs(pitch.median_f0(tone(700.0), SR, "singing") - 700.0) < 2.0


def test_an_unknown_mode_is_rejected():
    with pytest.raises(AudioError):
        pitch.median_f0(tone(220.0), SR, "humming")


# --- the suggestion -------------------------------------------------------


def test_an_octave_up_is_twelve_semitones():
    assert pitch.semitones_between(220.0, 440.0) == 12


def test_an_octave_down_is_minus_twelve():
    assert pitch.semitones_between(440.0, 220.0) == -12


def test_the_same_voice_needs_no_shift():
    assert pitch.semitones_between(200.0, 200.0) == 0


def test_the_suggestion_is_clamped_to_one_octave():
    """Plan §7 clamps before the per-mode limit narrows it further."""
    assert pitch.semitones_between(80.0, 900.0) == 12
    assert pitch.semitones_between(900.0, 80.0) == -12


def test_an_undetectable_side_means_no_shift():
    """0 is the one answer that is never actively wrong."""
    assert pitch.semitones_between(None, 220.0) == 0
    assert pitch.semitones_between(220.0, None) == 0
    assert pitch.semitones_between(None, None) == 0


def test_a_male_to_female_pair_lands_in_the_expected_band():
    """Plan §7 acceptance: the suggestion for male→female is +10 to +14."""
    male = pitch.median_f0(tone(130.0, 6.0), SR, "singing")
    female = pitch.median_f0(tone(245.0, 6.0), SR, "singing")
    assert 10 <= pitch.semitones_between(male, female) <= 14


def test_a_long_instrumental_intro_does_not_drag_the_answer():
    """Plan §7 acceptance: silence has to be excluded, not averaged in. The
    intro here is three times longer than the singing that follows."""
    intro = np.zeros(int(45 * SR), dtype=np.float32)
    with_intro = np.concatenate([intro, tone(220.0, 15.0)])
    assert abs(pitch.median_f0(with_intro, SR, "singing") - 220.0) < 1.0


def test_a_shift_survives_a_long_intro_on_the_source():
    intro = np.zeros(int(30 * SR), dtype=np.float32)
    source = np.concatenate([intro, tone(140.0, 15.0)])
    shift = pitch.semitones_between(
        pitch.median_f0(source, SR, "singing"),
        pitch.median_f0(tone(280.0, 6.0), SR, "singing"),
    )
    assert shift == 12


# --- batching -------------------------------------------------------------


def test_a_file_longer_than_one_block_is_analysed_whole(monkeypatch):
    """The FFT runs in blocks to bound memory; the seam between blocks must not
    change the answer."""
    monkeypatch.setattr(pitch, "BLOCK_FRAMES", 7)
    assert abs(pitch.median_f0(tone(220.0, 3.0), SR, "singing") - 220.0) < 1.0


def test_a_non_contiguous_input_is_framed_correctly():
    """`as_strided` reads raw strides, so a sliced array is the case that would
    quietly produce garbage instead of failing."""
    padded = np.repeat(tone(220.0, 2.0), 2)
    assert abs(pitch.median_f0(padded[::2], SR, "singing") - 220.0) < 1.0
