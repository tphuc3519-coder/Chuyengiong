"""Cleaning the voice sample, checked on signals whose answer is known.

Every test here builds audio where the right answer is arithmetic rather than
taste: a tone plus a measured amount of hiss, a tone plus a measured amount of
rumble, a clip whose first half is loud noise and second half is voice. What is
asserted is that the thing that should go down goes down by a lot and the thing
that should stay stays, which is all a spectral subtraction can honestly
promise and all that matters for what it is for.
"""

import numpy as np
import pytest

from modal_app import audio_utils as au
from modal_app import reference as ref

SR = 44100


def voice(seconds: float, freq: float = 140.0, amplitude: float = 0.3) -> np.ndarray:
    """A tone with a syllable-rate envelope. Loud, periodic, in bursts."""
    t = np.arange(int(seconds * SR)) / SR
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    return (amplitude * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def hiss(seconds: float, amplitude: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def rumble(seconds: float, freq: float = 25.0, amplitude: float = 0.05) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_energy(audio: np.ndarray, low: float, high: float) -> float:
    """RMS magnitude of the spectrum between `low` and `high` Hz."""
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    inside = spectrum[(freqs >= low) & (freqs < high)]
    return float(np.sqrt((inside**2).mean())) if inside.size else 0.0


# --- the STFT it is all built on ------------------------------------------


def test_an_unmodified_round_trip_returns_the_signal():
    """The one property everything else depends on. Analysis and synthesis are
    both root-Hann at half overlap, which multiplies to a Hann that sums to
    exactly one — so a spectrogram that is not touched has to come back as what
    went in, ends included."""
    clip = voice(3)
    spectrum, pad = ref._stft(clip)
    assert np.abs(ref._istft(spectrum, len(clip), pad) - clip).max() < 1e-5


def test_the_high_pass_mask_passes_speech_and_stops_the_room():
    mask = ref.highpass_mask(SR)
    freqs = np.fft.rfftfreq(ref.FFT_SIZE, 1.0 / SR)
    assert mask[freqs < ref.HIGHPASS_STOP_HZ].max() == pytest.approx(0.0, abs=1e-6)
    assert mask[freqs > 200].min() == pytest.approx(1.0, abs=1e-6)


# --- what it removes ------------------------------------------------------


def test_rumble_is_removed_and_the_voice_is_not():
    """The cheapest of the three corrections and the one with the most to gain:
    below 40 Hz there is nothing anybody said, and it is loud."""
    clip = voice(6) + rumble(6)
    cleaned = ref.denoise(clip, SR)
    assert band_energy(cleaned, 10, 40) < 0.05 * band_energy(clip, 10, 40)
    assert band_energy(cleaned, 120, 160) > 0.7 * band_energy(clip, 120, 160)


def test_the_noise_floor_is_estimated_and_taken_out():
    """The correction that matters most for how the conversion sounds: hiss in
    the reference is hiss fused into the converted timbre."""
    clip = voice(6) + hiss(6)
    cleaned = ref.denoise(clip, SR)
    assert band_energy(cleaned, 6000, 12000) < 0.6 * band_energy(clip, 6000, 12000)
    assert band_energy(cleaned, 120, 160) > 0.7 * band_energy(clip, 120, 160)


def test_nothing_is_gated_all_the_way_to_silence():
    """`SPECTRAL_FLOOR` is the difference between noise reduction and the
    burbling artefact it is famous for. A bin that goes to zero in one frame
    and back in the next is what makes that sound."""
    quiet = hiss(4, amplitude=0.01)
    cleaned = ref.denoise(quiet, SR)
    assert float(np.abs(cleaned).max()) > 0


def test_a_clip_too_short_to_analyse_is_handed_back_rather_than_mangled():
    tiny = voice(0.01)
    assert np.array_equal(ref.denoise(tiny, SR), tiny)


def test_digital_silence_survives_every_step():
    silence = np.zeros(int(2 * SR), dtype=np.float32)
    assert not np.any(np.isnan(ref.clean(silence, SR)))
    assert float(np.abs(ref.clean(silence, SR)).max()) == 0.0


# --- the level ------------------------------------------------------------


def test_a_quiet_sample_is_brought_up_and_a_loud_one_is_brought_down():
    quiet = ref.normalise(voice(4, amplitude=0.01))
    loud = ref.normalise(voice(4, amplitude=0.9))
    assert float(np.abs(quiet).max()) > 0.05
    assert float(np.abs(loud).max()) <= ref.MAX_PEAK


def test_the_level_is_measured_on_the_speech_and_not_on_the_silence():
    """Two recordings of the same voice, one with a long pause in it. An RMS
    over the whole file would scale them differently; the point of measuring
    only the loud samples is that it does not."""
    speech = voice(6)
    padded = np.concatenate([speech, np.zeros(int(10 * SR), dtype=np.float32)])
    assert float(np.abs(ref.normalise(speech)).max()) == pytest.approx(
        float(np.abs(ref.normalise(padded)).max()), rel=0.05
    )


def test_nothing_is_normalised_into_clipping():
    peaky = np.zeros(int(4 * SR), dtype=np.float32)
    peaky[::1000] = 0.99  # a click train: enormous crest factor, tiny RMS
    assert float(np.abs(ref.normalise(peaky)).max()) <= ref.MAX_PEAK


# --- which windows get used -----------------------------------------------


def test_loud_hiss_is_not_mistaken_for_somebody_talking():
    """A recording that opens with loud noise, which level alone scores as the
    best voice in it. The scorer lives in `audio_utils` and is shared with
    `usable_reference_window`, so both halves of the choice agree about what a
    voice is."""
    flags = au.speech_flags(np.concatenate([hiss(20, amplitude=0.5), voice(20)]))
    half = len(flags) // 2
    assert flags[:half].mean() == 0.0
    assert flags[half:].mean() > 0.4


def test_a_short_reference_yields_no_extra_windows():
    """Most references are one window long, and nothing about them changes."""
    main, extras = ref.prepare(voice(12), SR)
    assert extras == []
    assert len(main) == len(voice(12))


def test_a_long_reference_yields_more_than_one_window():
    clip = np.concatenate([voice(25), hiss(6, amplitude=0.004), voice(25)])
    main, extras = ref.prepare(clip, SR)
    assert len(main) / SR == pytest.approx(au.REFERENCE_MAX_SEC, abs=0.1)
    assert len(extras) == 1
    assert len(extras[0]) / SR == pytest.approx(au.REFERENCE_MAX_SEC, abs=0.1)


def test_the_extra_windows_do_not_overlap_the_main_one_or_each_other():
    """Averaging a window into an embedding it is already most of would only
    weight that stretch of the recording twice."""
    cleaned = ref.clean(voice(90), SR)
    taken = au.usable_reference_window(cleaned, SR)
    ranges = ref.extra_ranges(cleaned, SR, taken, ref.STYLE_WINDOWS - 1)
    assert len(ranges) == ref.STYLE_WINDOWS - 1
    for span in [taken, *ranges]:
        for other in ranges:
            if span is other:
                continue
            assert span[1] <= other[0] or other[1] <= span[0]


def test_a_stretch_of_room_tone_is_never_offered_as_an_extra_window():
    """`MIN_WINDOW_SPEECH`: an embedding averaged with a room is a worse
    estimate of the speaker than the one window it started from."""
    clip = np.concatenate([voice(25), hiss(45, amplitude=0.004)])
    _, extras = ref.prepare(clip, SR)
    assert extras == []


def test_asking_for_one_window_asks_for_the_old_behaviour():
    clip = np.concatenate([voice(25), voice(25, freq=200)])
    main, extras = ref.prepare(clip, SR, windows=1)
    assert extras == []
    assert len(main) / SR == pytest.approx(au.REFERENCE_MAX_SEC, abs=0.1)


def test_a_reference_that_is_too_short_is_still_refused():
    """The length rule lives in `audio_utils` and is called here, not copied."""
    with pytest.raises(au.AudioError):
        ref.prepare(voice(3), SR)


def test_a_recording_that_opens_with_a_loud_noise_is_still_cut_at_the_voice():
    """The case level alone gets wrong, end to end: 25 seconds of loud hiss and
    then somebody talking, quietly. What reaches the model has to be the
    talking."""
    clip = np.concatenate([hiss(25, amplitude=0.35), voice(25, amplitude=0.12)])
    main, _ = ref.prepare(clip, SR)
    # Periodic, not broadband: what came back is the tone, not the noise.
    assert band_energy(main, 120, 160) > 5 * band_energy(main, 6000, 12000)
