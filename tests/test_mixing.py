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
from modal_app.audio_utils import decode_audio, encode_wav, encode_wav_channels

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.2) -> bytes:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return encode_wav(amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32), SR)


def duration_of(data: bytes) -> float:
    return len(decode_audio(data, SR)) / SR


def stereo_tone(seconds: float, left: float = 220.0, right: float = 330.0) -> bytes:
    """A wav with a different tone per channel, so a downmix is visible."""
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    frames = np.stack(
        [0.2 * np.sin(2 * np.pi * left * t), 0.2 * np.sin(2 * np.pi * right * t)], axis=1
    )
    return encode_wav_channels(frames.astype(np.float32), SR)


def channels_of(data: bytes) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=nw=1:nk=1",
            "-",
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return int(out.stdout.decode().strip())


def bin_at(audio: np.ndarray, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    return float(spectrum[np.argmin(abs(freqs - freq))])


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


# --- the watermark hook ----------------------------------------------------
#
# The model needs torch and its own container, so what is testable here is the
# contract `mixing` offers it: what it is handed, when it is called, and what
# happens to what it gives back.


@needs_ffmpeg
def test_the_watermark_is_handed_readable_audio_not_an_mp3():
    """It runs on the mixed wav, after loudnorm and before the encode. Handing
    it the encoded mp3 would mean marking a file that is already final."""
    seen = []

    def watermark(data: bytes) -> bytes:
        seen.append(decode_audio(data, SR))
        return data

    mixing.mix(tone(1.0), tone(1.0, 440.0), watermark=watermark)
    assert len(seen) == 1
    assert len(seen[0]) == pytest.approx(SR, abs=SR * 0.15)


@needs_ffmpeg
def test_what_the_watermark_returns_is_what_gets_encoded():
    """The whole point: the shipped mp3 is the marked audio, not the input."""
    marked = tone(1.0, freq=880.0)
    out = decode_audio(mixing.mix(tone(1.0), tone(1.0, 440.0), watermark=lambda _: marked), SR)

    spectrum = np.abs(np.fft.rfft(out[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    peak = freqs[np.argmax(spectrum)]
    assert peak == pytest.approx(880.0, abs=10.0)


@needs_ffmpeg
def test_a_watermarked_output_is_still_tagged_ai_generated(tmp_path):
    out = mixing.mix(tone(1.0), tone(1.0), watermark=lambda data: data)
    assert comment_of(out, tmp_path) == mixing.AI_COMMENT


@needs_ffmpeg
def test_the_speech_branch_gets_the_same_hook():
    seen = []
    mixing.to_mp3(tone(1.0), watermark=lambda data: seen.append(data) or data)
    assert len(seen) == 1


@needs_ffmpeg
def test_a_watermark_that_produces_nothing_fails_the_job():
    """Rather than quietly shipping the unmarked mix: an output that claims to
    be watermarked and is not is worse than no output."""
    with pytest.raises(mixing.MixError):
        mixing.mix(tone(1.0), tone(1.0), watermark=lambda _: b"")


@needs_ffmpeg
def test_a_stereo_mix_reaches_the_watermark_as_stereo():
    """The watermark step reads the mix and writes it back. If it were handed a
    downmix, the song would come back mono."""
    seen = []
    mixing.mix(tone(1.0), stereo_tone(1.0), watermark=lambda data: seen.append(data) or data)
    assert channels_of(seen[0]) == 2


# --- channel layout --------------------------------------------------------


@needs_ffmpeg
def test_a_mono_vocal_does_not_drag_the_backing_track_down_to_mono():
    """`amix` settles on the narrowest layout across its inputs, and Seed-VC
    only ever returns mono — so without the `pan`, every song shipped with the
    instrumental's stereo image folded away."""
    out = mixing.mix(tone(1.0), stereo_tone(1.0, left=220.0, right=880.0))
    assert channels_of(out) == 2
    # The side that was only in the right channel is still there.
    left, right = _split(out)
    assert bin_at(right, 880.0) > 10 * bin_at(left, 880.0)


@needs_ffmpeg
def test_the_vocal_lands_in_both_channels_at_the_same_level():
    """`pan` rather than an `aformat` upmix: ffmpeg's mono-to-stereo conversion
    applies the -3 dB centre mix level, which would move the vocal down in the
    mix as a side effect of fixing the layout."""
    path_out = mixing.mix(tone(1.0, freq=220.0), stereo_tone(1.0, left=440.0, right=880.0))
    left, right = _split(path_out)
    assert bin_at(left, 220.0) == pytest.approx(bin_at(right, 220.0), rel=0.05)
    # ...and each side still carries only its own half of the backing track.
    assert bin_at(left, 440.0) > 10 * bin_at(left, 880.0)
    assert bin_at(right, 880.0) > 10 * bin_at(right, 440.0)


def _split(mp3: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "pipe:1",
        ],
        input=mp3,
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    return frames[:, 0].copy(), frames[:, 1].copy()
