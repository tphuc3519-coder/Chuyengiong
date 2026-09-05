"""The clarity chain, run through ffmpeg for real.

Two kinds of assertion here and they answer different questions. The string
tests answer "is this a filter graph ffmpeg will accept", which is worth
asking because the failure mode is a container dying on somebody's job rather
than anything visible at build time. The measured tests answer "does it do what
it says": rumble down, hiss down, the voice still there, and — the one that
matters most — nothing at all happening at zero.
"""

import shutil
import subprocess

import numpy as np
import pytest

from modal_app import enhance, mixing
from modal_app.audio_utils import decode_audio, encode_wav

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def voice(seconds: float = 4.0, freq: float = 200.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rumble(seconds: float = 4.0, amplitude: float = 0.08) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 22 * t)).astype(np.float32)


def hiss(seconds: float = 4.0, amplitude: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (amplitude * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def run(audio: np.ndarray, clarity: float) -> np.ndarray:
    """`audio` through the chain alone, with nothing else in the graph."""
    graph = f"[0:a]{enhance.chain(clarity, ',')}anull[out]"
    with_input = encode_wav(audio, SR)
    return decode_audio(
        _ffmpeg(with_input, graph),
        SR,
    )


def _ffmpeg(data: bytes, graph: str) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        out = Path(tmp) / "out.wav"
        src.write_bytes(data)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        return out.read_bytes()


def band_energy(audio: np.ndarray, low: float, high: float) -> float:
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    inside = spectrum[(freqs >= low) & (freqs < high)]
    return float(np.sqrt((inside**2).mean())) if inside.size else 0.0


# --- the contract at zero -------------------------------------------------


def test_zero_emits_no_filters_at_all():
    """The whole reason the amount is a slider and not a boolean: the output
    this app shipped before the chain existed has to still be reachable, and
    "reachable" means no filter ran, not that the filters were gentle."""
    assert enhance.filters(0) == []
    assert enhance.chain(0) == ""
    assert enhance.chain(0, ",") == ""


def test_the_comma_is_only_added_when_there_is_something_to_join():
    """A graph with a stray comma is an ffmpeg error at run time, in a
    container, on a job somebody is waiting for."""
    assert enhance.chain(1.0, ",").endswith(",")
    assert not enhance.chain(1.0, ",").endswith(",,")


def test_out_of_range_is_clamped_and_nonsense_falls_back():
    assert enhance.clamp_clarity(-1) == enhance.CLARITY_MIN
    assert enhance.clamp_clarity(99) == enhance.CLARITY_MAX
    assert enhance.clamp_clarity("loud") == enhance.DEFAULT_CLARITY
    assert enhance.clamp_clarity(None) == enhance.DEFAULT_CLARITY


def test_the_chain_scales_with_the_amount():
    """Every gain in it is a multiple of one number, so half is half."""
    assert "g=-2.50" in enhance.chain(1.0)
    assert "g=-1.25" in enhance.chain(0.5)


def test_the_denoiser_runs_before_everything_that_has_gain():
    """Order is not arrangeable: a lift applied before the denoiser amplifies
    the floor it is about to remove, and de-essing before the presence lift
    measures sibilance that has not happened yet."""
    names = [item.split("=")[0] for item in enhance.filters(1.0)]
    assert names.index("afftdn") < names.index("equalizer")
    assert names.index("equalizer") < names.index("deesser")
    assert names[0] == "aformat"


# --- what it actually does ------------------------------------------------


@needs_ffmpeg
def test_ffmpeg_accepts_the_graph_at_both_ends_of_the_slider():
    for clarity in (0.0, 0.25, 1.0):
        assert len(run(voice(), clarity)) > 0


@needs_ffmpeg
def test_nothing_is_done_to_the_audio_at_zero():
    clip = voice() + rumble()
    processed = run(clip, 0.0)
    assert band_energy(processed, 10, 40) == pytest.approx(band_energy(clip, 10, 40), rel=0.02)


@needs_ffmpeg
def test_rumble_goes_and_the_voice_stays():
    clip = voice() + rumble()
    processed = run(clip, 1.0)
    assert band_energy(processed, 10, 40) < 0.2 * band_energy(clip, 10, 40)
    assert band_energy(processed, 180, 220) > 0.5 * band_energy(clip, 180, 220)


@needs_ffmpeg
def test_the_noise_floor_comes_down():
    """A vocoder's haze is broadband and quiet, which is exactly what `afftdn`
    is for. Measured on hiss alone: with a tone in it the tracker has a signal
    to protect and reduces less, which is the correct behaviour and not a
    measurement of the filter."""
    clip = hiss(4.0, amplitude=0.05)
    processed = run(clip, 1.0)
    assert float(np.sqrt((processed**2).mean())) < 0.8 * float(np.sqrt((clip**2).mean()))


@needs_ffmpeg
def test_the_presence_band_comes_up_relative_to_the_mud():
    """The two peaking filters are the whole of what `trong hơn` means: less
    around 300 Hz, more around 3.4 kHz."""
    t = np.arange(int(4 * SR)) / SR
    both = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 3400 * t)).astype(
        np.float32
    )
    before = band_energy(both, 3300, 3500) / band_energy(both, 250, 350)
    after_audio = run(both, 1.0)
    after = band_energy(after_audio, 3300, 3500) / band_energy(after_audio, 250, 350)
    assert after > before * 1.3


# --- where it is wired in -------------------------------------------------


@needs_ffmpeg
def test_the_song_mix_only_enhances_the_vocal():
    """The instrumental is the separator's output and nothing here has any
    business filtering it — the chain sits on input 0, ahead of the `amix`."""
    assert "[0:a]" + enhance.chain(0.5, ",") in _mix_graph()


def _mix_graph() -> str:
    """The graph `mixing.mix` builds, captured without running ffmpeg."""
    captured = {}

    def fake(inputs, filter_complex, args, suffix):
        # The first call is the mix; `mix` calls again to encode the mp3, and
        # that second graph is an `anull` with nothing to say.
        captured.setdefault("graph", filter_complex)
        return encode_wav(voice(0.5), SR)

    original = mixing._ffmpeg
    mixing._ffmpeg = fake
    try:
        mixing.mix(encode_wav(voice(), SR), encode_wav(voice(), SR), clarity=0.5)
    finally:
        mixing._ffmpeg = original
    return captured["graph"]


@needs_ffmpeg
def test_a_speech_output_is_enhanced_too_and_zero_leaves_it_alone():
    quiet = encode_wav(voice() + rumble(), SR)
    plain = decode_audio(mixing.to_mp3(quiet, clarity=0), SR)
    cleaned = decode_audio(mixing.to_mp3(quiet, clarity=1.0), SR)
    assert band_energy(cleaned, 10, 40) < 0.5 * band_energy(plain, 10, 40)
