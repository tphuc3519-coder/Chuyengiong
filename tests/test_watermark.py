"""The watermark plumbing, minus the model.

AudioSeal needs torch and a checkpoint download, so what runs in CI is
everything around it: the on/off switch, the message encoding, and the signal
handling that decides whether a watermark survives the trip from a 16kHz mono
model to a 44.1kHz stereo mp3. That plumbing is where the bugs would be — the
model itself either loads or does not.
"""

import shutil
from itertools import pairwise

import numpy as np
import pytest

from modal_app import watermark as wm
from modal_app.app import MODEL_DIR

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# --- the switch -----------------------------------------------------------


def test_watermarking_is_on_unless_it_is_turned_off():
    """Default-on: a safety feature that defaults off is one nobody notices is
    broken. Empty means unset, which is the deploy workflow's normal state."""
    assert wm.enabled("") is True
    assert wm.enabled("   ") is True
    assert wm.enabled("1") is True
    assert wm.enabled("true") is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " False "])
def test_the_escape_hatch_takes_the_obvious_spellings(value):
    assert wm.enabled(value) is False


def test_the_flag_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(wm.ENV_FLAG, "0")
    assert wm.enabled() is False
    monkeypatch.delenv(wm.ENV_FLAG)
    assert wm.enabled() is True


# --- the message ----------------------------------------------------------


def test_the_message_round_trips():
    bits = wm.message_bits()
    assert len(bits) == wm.MESSAGE_BITS == 16
    assert set(bits) <= {0, 1}
    assert wm.bits_to_int(bits) == wm.WATERMARK_MESSAGE


def test_the_message_is_not_a_degenerate_pattern():
    """All-zeros and all-ones are what an unwatermarked file is most likely to
    decode to, which would make `ours` fire on noise."""
    bits = wm.message_bits()
    assert 0 in bits and 1 in bits


def test_a_message_that_does_not_fit_is_refused():
    with pytest.raises(wm.WatermarkError):
        wm.message_bits(1 << 16)
    with pytest.raises(wm.WatermarkError):
        wm.message_bits(-1)


def test_message_bits_are_most_significant_first():
    assert wm.message_bits(1, nbits=4) == [0, 0, 0, 1]
    assert wm.message_bits(8, nbits=4) == [1, 0, 0, 0]


# --- windowing ------------------------------------------------------------


def test_a_short_signal_is_one_window():
    assert wm.window_bounds(1000, 4000, 100) == [(0, 1000)]
    assert wm.window_bounds(4000, 4000, 100) == [(0, 4000)]


def test_windows_cover_everything_and_overlap_by_the_right_amount():
    bounds = wm.window_bounds(10_000, 4000, 100)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10_000
    for (_, end), (next_start, _) in pairwise(bounds):
        assert end - next_start == 100
    assert all(end - start <= 4000 for start, end in bounds)


def test_no_window_is_longer_than_the_model_gets_fed():
    """The reason windowing exists: one pass over an 8 minute file allocates
    gigabytes in the SEANet stack."""
    eight_minutes = 8 * 60 * wm.MODEL_SAMPLE_RATE
    window = int(wm.WINDOW_SEC * wm.MODEL_SAMPLE_RATE)
    bounds = wm.window_bounds(eight_minutes, window, 100)
    assert max(end - start for start, end in bounds) <= window
    assert bounds[-1][1] == eight_minutes


def test_impossible_windowing_is_refused():
    with pytest.raises(wm.WatermarkError):
        wm.window_bounds(0, 4000, 100)
    with pytest.raises(wm.WatermarkError):
        wm.window_bounds(10_000, 100, 100)  # overlap swallows the window


# --- joining --------------------------------------------------------------


def _windowed(signal: np.ndarray, bounds: list[tuple[int, int]]) -> list[np.ndarray]:
    return [signal[start:end] for start, end in bounds]


def test_joining_windows_of_one_signal_returns_that_signal():
    """The property that matters: a linear fade between two views of the same
    signal reconstructs it exactly. An equal-power fade would bump every join."""
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(10_000).astype(np.float32) * 0.01
    bounds = wm.window_bounds(len(signal), 4000, 200)
    joined = wm.blend(_windowed(signal, bounds), bounds, len(signal))
    assert joined.shape == signal.shape
    assert np.allclose(joined, signal, atol=1e-6)


def test_a_constant_signal_keeps_its_level_across_every_join():
    """The audible failure mode, stated directly: no dip and no bump at a seam."""
    signal = np.full(10_000, 0.05, dtype=np.float32)
    bounds = wm.window_bounds(len(signal), 4000, 200)
    joined = wm.blend(_windowed(signal, bounds), bounds, len(signal))
    assert np.allclose(joined, 0.05, atol=1e-6)


def test_a_short_window_is_padded_rather_than_shifting_the_rest():
    bounds = wm.window_bounds(10_000, 4000, 200)
    parts = [np.ones(end - start, dtype=np.float32) for start, end in bounds]
    parts[0] = parts[0][:-500]  # a model that returned a short tail
    joined = wm.blend(parts, bounds, 10_000)
    assert len(joined) == 10_000
    # The gap is filled with silence, and the window after it still lands where
    # its bounds say it does.
    assert joined[-1] == pytest.approx(1.0)


def test_a_window_count_mismatch_is_an_error_not_a_silent_truncation():
    bounds = wm.window_bounds(10_000, 4000, 200)
    with pytest.raises(wm.WatermarkError):
        wm.blend([np.zeros(10)], bounds, 10_000)


# --- resampling and fitting -----------------------------------------------


def test_fit_pads_and_trims():
    assert len(wm.fit(np.zeros(10, dtype=np.float32), 15)) == 15
    assert len(wm.fit(np.zeros(20, dtype=np.float32), 15)) == 15


@needs_ffmpeg
def test_resampling_is_a_no_op_at_the_same_rate():
    signal = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    assert np.array_equal(wm.resample(signal, 16000, 16000), signal)


@needs_ffmpeg
def test_resampling_keeps_the_length_and_the_level():
    t = np.arange(16000, dtype=np.float32) / 16000
    signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = wm.resample(signal, 16000, 44100)
    assert abs(len(out) - 44100) < 200
    rms = float(np.sqrt((out**2).mean()))
    assert rms == pytest.approx(float(np.sqrt((signal**2).mean())), rel=0.05)


@needs_ffmpeg
def test_a_quiet_watermark_survives_the_resampler_intact():
    """The regression this guards: a watermark peaks tens of dB below the music,
    and a 16-bit round trip at its own scale would quantise it into noise. The
    resampler normalises before the round trip and restores the scale after."""
    t = np.arange(16000, dtype=np.float32) / 16000
    quiet = (0.001 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = wm.resample(quiet, 16000, 44100)
    rms = float(np.sqrt((out**2).mean()))
    assert rms == pytest.approx(float(np.sqrt((quiet**2).mean())), rel=0.05)
    # Signal well clear of what is left of the noise floor.
    assert rms > 20 * float(np.abs(out[:50]).mean() + 1e-12) or rms > 5e-4


@needs_ffmpeg
def test_resampling_silence_does_not_divide_by_zero():
    out = wm.resample(np.zeros(16000, dtype=np.float32), 16000, 44100)
    assert not np.any(out)
    assert abs(len(out) - 44100) < 200


# --- container definition -------------------------------------------------


def test_audioseal_is_pinned_to_the_version_this_was_written_against():
    """0.2 is the release that stopped resampling internally, which is why this
    module resamples. An older one would silently do it twice."""
    assert wm.AUDIOSEAL_SPEC == "audioseal==0.2.0"


def test_checkpoints_are_cached_on_the_model_volume():
    """`AUDIOSEAL_CACHE_DIR` is the first variable audioseal's loader checks. If
    it did not point at the Volume, every cold container would re-download.

    A Modal Image does not expose its build steps, so this asserts on the
    constant the builder is given — the same trick `test_deploy` uses."""
    assert wm.CACHE_ENV == "AUDIOSEAL_CACHE_DIR"
    assert MODEL_DIR == "/models"


def test_the_model_rate_is_the_one_the_checkpoints_were_trained_at():
    assert wm.MODEL_SAMPLE_RATE == 16000
