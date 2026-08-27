"""The separation port, checked where it can be checked without a GPU.

The model itself needs an A10G, so what is testable here is the part that has
actually broken in the app this was ported from: output files whose names do
not match what the caller asks for, and an image that quietly lacks cuDNN and
falls back to CPU.
"""

import pytest

from modal_app import separation as sep


def test_the_default_is_the_two_stem_quality_model():
    assert sep.DEFAULT_SEPARATION_MODEL == "roformer"
    cfg = sep.SEPARATION_MODELS["roformer"]
    # The checkpoint name is audio-separator's lookup key, carried over from
    # the app this was ported from. Renaming it silently loads nothing.
    assert cfg["file"] == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert cfg["stems"] == ["Vocals", "Instrumental"]


def test_every_model_produces_a_vocal_stem():
    for name, cfg in sep.SEPARATION_MODELS.items():
        assert sep.VOCAL_STEM in [s.lower() for s in cfg["stems"]], name


def test_unknown_model_is_refused_by_name():
    assert sep.check_model("htdemucs") == "htdemucs"
    with pytest.raises(sep.SeparationError):
        sep.check_model("roformer-v2")


def test_the_image_keeps_cudnn_and_the_gpu_runtime():
    """onnxruntime-gpu needs libcudnn.so.9; debian_slim has none and the model
    silently falls back to CPU, where it is tens of times slower."""
    assert "cudnn" in sep.CUDA_IMAGE_TAG
    assert "[gpu]" in sep.AUDIO_SEPARATOR_SPEC


# --- extension handling ---------------------------------------------------


def test_known_extensions_survive_and_unknown_ones_become_mp3():
    assert sep.safe_ext("song.m4a") == ".m4a"
    assert sep.safe_ext("SONG.FLAC") == ".flac"
    assert sep.safe_ext("recording") == sep.DEFAULT_EXT
    assert sep.safe_ext("virus.exe") == sep.DEFAULT_EXT
    assert sep.safe_ext(None) == sep.DEFAULT_EXT


def test_extracting_an_extension_twice_gives_the_same_answer():
    """The API extracts it from the upload, the pipeline re-checks the stored
    value: the second pass sees `.m4a`, not `song.m4a`."""
    for name in ("song.m4a", "clip.WAV", "nothing"):
        once = sep.safe_ext(name)
        assert sep.safe_ext(once) == once


# --- output naming --------------------------------------------------------


def write(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_decorated_model_output_is_renamed_to_the_stem(tmp_path):
    """What BS-Roformer actually writes when custom_output_names does not take."""
    out = write(
        tmp_path,
        "input_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav",
        "input_(Instrumental)_model_bs_roformer_ep_317_sdr_12.wav",
    )
    assert sep._collect_stems(out, ["Vocals", "Instrumental"]) == {
        "vocals": "vocals.wav",
        "instrumental": "instrumental.wav",
    }
    assert (out / "vocals.wav").is_file()


def test_already_correct_names_are_left_alone(tmp_path):
    out = write(tmp_path, "vocals.wav", "instrumental.wav")
    assert set(sep._collect_stems(out, ["Vocals", "Instrumental"])) == {
        "vocals",
        "instrumental",
    }


def test_unrecognisable_names_are_assigned_in_declaration_order(tmp_path):
    """Better a positional guess than failing a paid job over a naming scheme."""
    out = write(tmp_path, "0.wav", "1.wav")
    produced = sep._collect_stems(out, ["Vocals", "Instrumental"])
    assert produced == {"vocals": "vocals.wav", "instrumental": "instrumental.wav"}


def test_non_audio_files_are_ignored(tmp_path):
    out = write(tmp_path, "vocals.wav", "separator.log")
    assert sep._collect_stems(out, ["Vocals", "Instrumental"]) == {"vocals": "vocals.wav"}


def test_four_stem_output_keeps_all_four(tmp_path):
    out = write(tmp_path, "vocals.wav", "drums.wav", "bass.wav", "other.wav")
    produced = sep._collect_stems(out, sep.SEPARATION_MODELS["htdemucs"]["stems"])
    assert list(produced) == ["vocals", "drums", "bass", "other"]


# --- where the stems land -------------------------------------------------


class FakeInstance:
    """Stands in for the per-architecture separator `load_model` builds."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


class FakeSeparator:
    def __init__(self, output_dir, instance=True):
        self.output_dir = output_dir
        self.model_instance = FakeInstance(output_dir) if instance else None


def test_redirecting_output_moves_the_architecture_instance_too():
    """The bug this exists for: `load_model` runs once per container with
    MODEL_DIR and copies it into the instance's config, so moving only the
    wrapper leaves every stem in MODEL_DIR. The job then fails as "produced no
    vocal stem" while the model is working perfectly."""
    separator = FakeSeparator("/models")
    sep.point_output_at(separator, "/tmp/job/stems")
    assert separator.output_dir == "/tmp/job/stems"
    assert separator.model_instance.output_dir == "/tmp/job/stems"


def test_redirecting_output_before_a_model_is_loaded_is_not_an_error():
    """`model_instance` is None until `load_model` runs."""
    separator = FakeSeparator("/models", instance=False)
    sep.point_output_at(separator, "/tmp/job/stems")
    assert separator.output_dir == "/tmp/job/stems"
