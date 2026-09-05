"""The parts of voice training that do not need a GPU.

Which is: the dataset. Everything after it is upstream's `Trainer` and cannot
be tested without an A10G and ten minutes, so what is worth covering here is
the half that decides what the GPU is shown — the clip lengths `ft_dataset`
silently skips outside of, the silence that would otherwise teach the model
that this speaker is a room, and the refusal to spend the GPU at all on a
recording too short to learn anybody from.
"""

import numpy as np
import pytest

from modal_app import training
from modal_app.audio_utils import decode_wav

SR = 22050


def speech(seconds: float, gap_every: float = 0.0) -> np.ndarray:
    """A tone in syllable-length bursts, optionally with silences in it."""
    t = np.arange(int(seconds * SR)) / SR
    audio = 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) * np.sin(2 * np.pi * 150 * t)
    if gap_every:
        period = int(gap_every * SR)
        for start in range(period, len(audio), period):
            audio[start : start + int(0.4 * SR)] = 0.0
    return audio.astype(np.float32)


def test_steps_are_clamped_to_a_range_that_adapts_without_memorising():
    assert training.clamp_steps(None) == training.DEFAULT_STEPS
    assert training.clamp_steps(0) == training.DEFAULT_STEPS
    assert training.clamp_steps(1) == training.MIN_STEPS
    assert training.clamp_steps(10**6) == training.MAX_STEPS
    assert training.clamp_steps("many") == training.DEFAULT_STEPS


def test_every_clip_lands_inside_the_window_the_dataset_will_accept():
    """`data/ft_dataset.py` skips anything under 1s or over 30s without an
    error, so a dataset of the wrong lengths trains on nothing and says so
    nowhere."""
    clips = training.split_clips(speech(90, gap_every=6), SR)
    assert clips
    for clip in clips:
        assert training.CLIP_MIN_SEC <= len(clip) / SR <= training.CLIP_MAX_SEC + 0.5


def test_silence_is_not_written_into_the_dataset():
    """A stretch of nothing teaches the model that this speaker is a room."""
    quiet = np.zeros(int(30 * SR), dtype=np.float32)
    assert training.split_clips(quiet, SR) == []


def test_a_dataset_is_written_as_wavs_at_the_configs_own_rate(tmp_path):
    total = training.build_dataset([speech(120, gap_every=6)], SR, str(tmp_path))
    written = sorted(tmp_path.glob("*.wav"))
    assert written
    assert total > training.MIN_TOTAL_SEC
    audio, rate = decode_wav(written[0].read_bytes())
    assert rate == SR
    assert training.CLIP_MIN_SEC <= len(audio) / SR <= training.CLIP_MAX_SEC + 0.5


def test_too_little_audio_is_refused_before_the_gpu_is_booked(tmp_path):
    """Ten GPU minutes spent memorising one sentence is the failure this
    prevents, and the person who started the run is the only one who can fix
    it — by recording more."""
    with pytest.raises(training.TrainingError):
        training.build_dataset([speech(8)], SR, str(tmp_path))


def test_the_dataset_is_capped_so_a_long_recording_does_not_run_forever(tmp_path):
    total = training.build_dataset([speech(60, gap_every=5)] * 40, SR, str(tmp_path))
    assert total <= training.MAX_TOTAL_SEC + training.CLIP_MAX_SEC


def test_every_mode_that_can_be_converted_can_also_be_trained():
    """A profile is per mode, and a mode with no config is a mode nobody can
    train a voice for — which would be a surprise found at run time."""
    from modal_app.audio_utils import MODES

    assert set(training.CONFIGS) == set(MODES)
    assert set(training.PRETRAINED) == set(MODES)
