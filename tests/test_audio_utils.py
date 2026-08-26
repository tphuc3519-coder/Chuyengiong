"""Tests for the chunking / crossfade / validation rules of Phase 1.

These run without a GPU, without Modal and without seed-vc: they cover the part
of `convert()` that is ours rather than upstream's, which is also the part the
plan calls out as the easiest to get wrong.
"""

import shutil

import numpy as np
import pytest

from modal_app import audio_utils as au

SR = 8000  # low rate keeps the fixtures small; nothing here is rate-specific


def noise(seconds: float, sample_rate: int = SR, seed: int = 0, level: float = 0.3):
    """Uniform noise, bounded well inside full scale so nothing clips."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1.0, 1.0, int(seconds * sample_rate)) * level).astype(np.float32)


# --- wav bytes ------------------------------------------------------------


def test_wav_round_trip():
    audio = noise(0.5)
    decoded, rate = au.decode_wav(au.encode_wav(audio, SR))
    assert rate == SR
    assert len(decoded) == len(audio)
    # 16-bit quantisation is the only loss.
    assert np.max(np.abs(decoded - audio)) < 1e-4


def test_encode_wav_clips_instead_of_wrapping():
    decoded, _ = au.decode_wav(au.encode_wav(np.array([2.0, -2.0], dtype=np.float32), SR))
    assert decoded[0] > 0.99
    assert decoded[1] < -0.99


def test_decode_wav_rejects_non_wav():
    with pytest.raises(au.AudioError):
        au.decode_wav(b"definitely not a wav file")


def test_encode_wav_rejects_stereo():
    with pytest.raises(au.AudioError):
        au.encode_wav(np.zeros((2, 10), dtype=np.float32), SR)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_audio_resamples_to_target_rate():
    audio = decode = au.decode_audio(au.encode_wav(noise(1.0), SR), 16000)
    assert isinstance(decode, np.ndarray)
    assert abs(len(audio) - 16000) < 200  # 1 second at the requested rate


def test_decode_audio_rejects_empty_input():
    with pytest.raises(au.AudioError):
        au.decode_audio(b"", SR)


# --- validation -----------------------------------------------------------


def test_reference_shorter_than_five_seconds_is_rejected():
    with pytest.raises(au.AudioError, match="at least"):
        au.prepare_reference(noise(3.0), SR)


def test_reference_is_truncated_to_the_context_window():
    prepared = au.prepare_reference(noise(45.0), SR)
    assert au.duration_sec(prepared, SR) == pytest.approx(au.REFERENCE_MAX_SEC)


def test_reference_of_usable_length_is_untouched():
    reference = noise(12.0)
    assert len(au.prepare_reference(reference, SR)) == len(reference)


def test_source_over_the_limit_is_rejected():
    long_source = np.zeros(int((au.SOURCE_MAX_SEC + 1) * SR), dtype=np.float32)
    with pytest.raises(au.AudioError, match="minutes"):
        au.check_source(long_source, SR)


def test_empty_source_is_rejected():
    with pytest.raises(au.AudioError):
        au.check_source(np.zeros(0, dtype=np.float32), SR)


def test_semitone_shift_limits_differ_per_mode():
    assert au.clamp_semitone_shift(20, "singing") == 12
    assert au.clamp_semitone_shift(20, "speech") == 8
    assert au.clamp_semitone_shift(-20, "speech") == -8
    assert au.clamp_semitone_shift(3, "singing") == 3


def test_diffusion_steps_default_per_mode_and_clamp():
    assert au.clamp_diffusion_steps(0, "speech") == 25
    assert au.clamp_diffusion_steps(0, "singing") == 50
    assert au.clamp_diffusion_steps(5, "singing") == au.DIFFUSION_STEPS_MIN
    assert au.clamp_diffusion_steps(500, "singing") == au.DIFFUSION_STEPS_MAX
    assert au.clamp_diffusion_steps(37, "singing") == 37


def test_unknown_mode_is_rejected():
    with pytest.raises(au.AudioError):
        au.check_mode("karaoke")


# --- chunking -------------------------------------------------------------


def test_short_audio_is_one_chunk():
    audio = noise(35.0)
    chunks = au.split_at_silence(audio, SR)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], audio)


def test_empty_audio_cannot_be_split():
    with pytest.raises(au.AudioError):
        au.split_at_silence(np.zeros(0, dtype=np.float32), SR)


def test_cut_lands_in_the_silent_gap():
    audio = noise(70.0)
    gap = slice(int(28.5 * SR), int(30.5 * SR))
    audio[gap] = 0.0

    chunks = au.split_at_silence(audio, SR)
    assert len(chunks) >= 2
    # The first boundary is where chunk 2 starts, plus the overlap it got back.
    cut = len(chunks[0]) + int(au.CHUNK_OVERLAP_SEC * SR)
    assert 28.5 * SR <= cut <= 30.5 * SR


def test_chunks_respect_the_length_bounds():
    chunks = au.split_at_silence(noise(8 * 60.0), SR)
    assert len(chunks) > 10
    for chunk in chunks:
        assert au.duration_sec(chunk, SR) <= au.CHUNK_MAX_SEC + au.CHUNK_OVERLAP_SEC + 0.01
    for chunk in chunks[:-1]:
        assert au.duration_sec(chunk, SR) >= au.CHUNK_MIN_SEC


def test_consecutive_chunks_share_the_overlap():
    overlap = int(au.CHUNK_OVERLAP_SEC * SR)
    chunks = au.split_at_silence(noise(120.0), SR)
    assert len(chunks) > 2
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert np.array_equal(previous[-overlap:], following[:overlap])


def test_short_tail_is_merged_into_the_previous_chunk():
    # Silence at 30s pulls the cut there; with a 15s minimum the 12s tail that
    # would leave cannot stand on its own, so it folds back in.
    audio = noise(42.0)
    audio[int(29.8 * SR) : int(30.2 * SR)] = 0.0
    chunks = au.split_at_silence(audio, SR, min_sec=15.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == len(audio)


# --- crossfade ------------------------------------------------------------


def test_crossfade_concat_restores_the_original_length():
    audio = noise(150.0)
    chunks = au.split_at_silence(audio, SR)
    assert len(chunks) > 2
    joined = au.crossfade_concat(chunks, SR)
    assert len(joined) == len(audio)


def test_crossfade_leaves_material_outside_the_joins_alone():
    audio = noise(150.0)
    chunks = au.split_at_silence(audio, SR)
    joined = au.crossfade_concat(chunks, SR)
    overlap = int(au.CHUNK_OVERLAP_SEC * SR)
    head = len(chunks[0]) - overlap
    assert np.allclose(joined[:head], audio[:head])


def test_crossfade_holds_level_across_the_join():
    """Equal power: uncorrelated halves must not dip in the middle of a join."""
    overlap = int(au.CHUNK_OVERLAP_SEC * SR)
    left = noise(5.0, seed=1)
    right = noise(5.0, seed=2)
    joined = au.crossfade_concat([left, right], SR)

    seam = len(left) - overlap
    inside = joined[seam : seam + overlap]
    outside = joined[seam - 4 * overlap : seam]
    ratio = float(np.sqrt(np.mean(inside**2)) / np.sqrt(np.mean(outside**2)))
    assert 0.85 < ratio < 1.15


def test_crossfade_handles_a_single_chunk_and_short_chunks():
    single = noise(1.0)
    assert np.array_equal(au.crossfade_concat([single], SR), single)

    tiny = noise(0.05)  # shorter than the 200ms overlap
    joined = au.crossfade_concat([single, tiny], SR)
    assert len(joined) == len(single) + len(tiny) - len(tiny)


def test_crossfade_needs_at_least_one_chunk():
    with pytest.raises(au.AudioError):
        au.crossfade_concat([], SR)


# --- rms ------------------------------------------------------------------


def test_frame_rms_matches_manual_computation():
    audio = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(au.frame_rms(audio, 2), [1.0, 0.0])
    assert len(au.frame_rms(np.zeros(1, dtype=np.float32), 4)) == 0
