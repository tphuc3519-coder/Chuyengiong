"""The ffmpeg filter graphs, run for real.

These need ffmpeg — CI installs it, so the mix is exercised rather than
described. What is worth asserting is not "ffmpeg ran": it is the three
decisions that make a mix sound wrong when they are missed. A mix that is 6 dB
quieter than its inputs means `normalize=0` was dropped; a mix as long as the
shorter input means `duration=longest` was; a file with no comment tag means an
output left our hands without saying it was AI-generated.
"""

import shutil
import subprocess

import numpy as np
import pytest

from modal_app import mixing
from modal_app.audio_utils import decode_audio, encode_wav

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.2) -> bytes:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return encode_wav(amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32), SR)


def duration_of(data: bytes) -> float:
    return len(decode_audio(data, SR)) / SR


def comment_of(data: bytes, tmp_path) -> str:
    path = tmp_path / "probe.mp3"
    path.write_bytes(data)
    out = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "format_tags=comment",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode().strip()


# --- gain clamping (no ffmpeg needed) -------------------------------------


def test_gain_is_clamped_to_the_usable_range():
    assert mixing.clamp_gain_db(40) == mixing.MAX_VOCAL_GAIN_DB
    assert mixing.clamp_gain_db(-40) == -mixing.MAX_VOCAL_GAIN_DB
    assert mixing.clamp_gain_db(3.5) == 3.5


def test_a_missing_or_unparseable_gain_is_no_gain():
    assert mixing.clamp_gain_db(None) == 0.0
    assert mixing.clamp_gain_db("loud") == 0.0


# --- the real thing --------------------------------------------------------


@needs_ffmpeg
def test_mix_runs_to_the_longer_input(tmp_path):
    """A converted vocal is never exactly as long as the instrumental it came
    from; truncating to the shorter one would cut the end off the song."""
    out = mixing.mix(tone(1.0), tone(2.0, freq=440.0))
    assert duration_of(out) == pytest.approx(2.0, abs=0.15)


@needs_ffmpeg
def test_mix_does_not_halve_the_level():
    """`amix` defaults to normalize=1, which divides by the input count."""
    quiet = decode_audio(mixing.mix(tone(2.0), tone(2.0, freq=440.0)), SR)
    assert float(np.abs(quiet).max()) > 0.3


@needs_ffmpeg
def test_vocal_gain_changes_the_balance():
    loud = decode_audio(mixing.mix(tone(2.0), tone(2.0, 440.0), vocal_gain_db=6), SR)
    soft = decode_audio(mixing.mix(tone(2.0), tone(2.0, 440.0), vocal_gain_db=-6), SR)

    # loudnorm pulls both to the same level, so compare the vocal's share of it:
    # the 220 Hz bin against the 440 Hz one.
    def share(audio):
        spectrum = np.abs(np.fft.rfft(audio[:SR]))
        freqs = np.fft.rfftfreq(SR, 1 / SR)
        return spectrum[np.argmin(abs(freqs - 220))] / spectrum[np.argmin(abs(freqs - 440))]

    assert share(loud) > share(soft)


@needs_ffmpeg
def test_every_output_is_tagged_ai_generated(tmp_path):
    assert comment_of(mixing.mix(tone(1.0), tone(1.0)), tmp_path) == mixing.AI_COMMENT
    assert comment_of(mixing.to_mp3(tone(1.0)), tmp_path) == mixing.AI_COMMENT


@needs_ffmpeg
def test_speech_output_is_an_mp3_of_the_same_length():
    assert duration_of(mixing.to_mp3(tone(1.5))) == pytest.approx(1.5, abs=0.15)


@needs_ffmpeg
def test_summing_four_stems_gives_one_instrumental():
    summed = mixing.sum_stems([tone(1.0, 220.0), tone(1.0, 440.0), tone(1.0, 660.0)])
    assert duration_of(summed) == pytest.approx(1.0, abs=0.1)


def test_summing_one_stem_is_a_no_op():
    """Cheap path for the 2-stem models, and it must not need ffmpeg."""
    data = tone(0.1)
    assert mixing.sum_stems([data]) is data


def test_empty_input_is_rejected_before_ffmpeg_sees_it():
    with pytest.raises(mixing.MixError):
        mixing.mix(b"", tone(0.1))
    with pytest.raises(mixing.MixError):
        mixing.sum_stems([])
