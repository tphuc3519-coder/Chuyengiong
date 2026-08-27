"""Mixdown and final encode, done with ffmpeg.

Plain subprocess calls, no numpy: this is the one step where a filter graph
beats array arithmetic, and keeping it dependency-free means the separation
container (which has ffmpeg but not our audio stack) can reuse `sum_stems`.

Three rules, all from the plan:

* `amix=normalize=0` — the default normalize=1 divides every input by the
  number of inputs, which drops a two-input mix by 6 dB for no reason;
* `loudnorm=I=-14` last, so the mix lands on the streaming reference level
  instead of wherever the sum happened to end up;
* an `AI-generated` comment tag on every output we hand back (Phase 6 item 2,
  done here because this is the only place output bytes are created).

`mix` and `to_mp3` take an optional `watermark` callable, applied to the
normalised wav in between the mix and the encode (plan §8, "Cân nhắc thêm").
It is a callable rather than an import because the model behind it needs torch
and its own container: this module stays plain ffmpeg, which is what lets the
separation container reuse `sum_stems` and lets CI test the whole path with a
stand-in.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

# -14 LUFS with 1 dB of headroom: the streaming reference, and the right target
# for something that will be listened to on a phone.
LOUDNORM = "loudnorm=I=-14:TP=-1.0"
# The mono vocal, copied to both channels at unity. See `mix`.
CENTRE = "pan=stereo|c0=c0|c1=c0"
OUTPUT_BITRATE = "192k"
# Beyond this the vocal is either buried or clipping the mix; the slider in the
# UI has no business going further.
MAX_VOCAL_GAIN_DB = 12.0
AI_COMMENT = "AI-generated voice conversion"


class MixError(RuntimeError):
    """ffmpeg refused to produce output."""


def clamp_gain_db(gain_db: float) -> float:
    try:
        value = float(gain_db)
    except (TypeError, ValueError):
        return 0.0
    return max(-MAX_VOCAL_GAIN_DB, min(MAX_VOCAL_GAIN_DB, value))


def _sample_rate(wav: bytes) -> int:
    """The rate of a wav this app wrote itself.

    `wave` and not ffprobe because it is stdlib and this module stays free of
    everything else; safe because the only wav reaching here is one `encode_wav`
    produced, which is plain PCM by construction.
    """
    if not wav:
        raise MixError("the vocal is empty")
    try:
        with wave.open(io.BytesIO(wav), "rb") as src:
            return src.getframerate()
    except (wave.Error, EOFError) as exc:  # truncated headers raise EOFError
        raise MixError(f"cannot read the vocal's sample rate: {exc}") from exc


def _ffmpeg(inputs: list[bytes], filter_complex: str, args: list[str], suffix: str) -> bytes:
    """Run one filter graph over N in-memory inputs and return the output file.

    Files rather than pipes on both ends: the mp3 muxer wants to seek back and
    write its header, and a wav input needs a size in its header before ffmpeg
    will treat it as anything but a stream.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for i, data in enumerate(inputs):
            if not data:
                raise MixError(f"input {i} is empty")
            path = tmpdir / f"in{i}.wav"
            path.write_bytes(data)
            cmd += ["-i", str(path)]
        out = tmpdir / f"out{suffix}"
        cmd += ["-filter_complex", filter_complex, "-map", "[out]", *args, str(out)]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not out.is_file():
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise MixError(f"ffmpeg failed: {detail[-1] if detail else 'no output'}")
        return out.read_bytes()


def _mp3_args() -> list[str]:
    return [
        "-c:a",
        "libmp3lame",
        "-b:a",
        OUTPUT_BITRATE,
        "-metadata",
        f"comment={AI_COMMENT}",
    ]


def _apply(mixed_wav: bytes, watermark: Callable[[bytes], bytes] | None) -> bytes:
    """Watermark if asked, then encode. The one place that order is decided.

    After `loudnorm`, never before: loudnorm applies a gain and a limiter, and
    a watermark added ahead of it would be rescaled by both. Before the encode,
    so what ships is what was marked.

    The encode is a separate ffmpeg pass even when nothing is watermarked. It
    could be folded back into the mix graph in that case, but then turning
    `WATERMARK` off would exercise a different code path than the one we test —
    and the cost of the extra pass is one 16-bit round trip of a file that came
    from 16-bit stems.
    """
    if watermark is None:
        return encode_mp3(mixed_wav)
    marked = watermark(mixed_wav)
    if not marked:
        raise MixError("watermarking returned no audio")
    return encode_mp3(marked)


def encode_mp3(audio_wav: bytes) -> bytes:
    """Tag and encode, no filtering. `anull` because `_ffmpeg` maps `[out]`."""
    return _ffmpeg([audio_wav], "[0:a]anull[out]", _mp3_args(), ".mp3")


def mix(
    vocal_wav: bytes,
    instrumental_wav: bytes,
    vocal_gain_db: float = 0.0,
    watermark: Callable[[bytes], bytes] | None = None,
) -> bytes:
    """Converted vocal over the original instrumental -> mp3 bytes.

    The two inputs are rarely the same length — conversion resamples and joins
    chunks — so `amix` runs to the longer of the two rather than truncating the
    song at whichever stem ends first.

    The `pan` is not cosmetic. `amix` negotiates one channel layout across its
    inputs and settles on the narrowest, so a mono vocal — and Seed-VC only
    ever returns mono — silently folded the instrumental's stereo image down
    with it. Copying the vocal to both channels first leaves the backing track
    as wide as it arrived. `pan` rather than an `aformat` upmix because
    ffmpeg's mono-to-stereo conversion applies the -3 dB centre mix level,
    which would quietly move the vocal back down in the mix.
    """
    gain = clamp_gain_db(vocal_gain_db)
    # `loudnorm` runs at 192 kHz internally and hands that rate on, so the wav
    # coming out is four times the size it needs to be and — because ffmpeg
    # describes a rate that high with an explicit channel layout — carries a
    # WAVE_FORMAT_EXTENSIBLE header. The watermark step then could not read its
    # own input ("unknown format: 65534") and the job died one stage from done.
    # Putting the rate back settles both: `amix` had already agreed on the
    # vocal's, which is what this restores.
    rate = _sample_rate(vocal_wav)
    graph = (
        f"[0:a]volume={gain:.2f}dB,{CENTRE}[v];"
        f"[v][1:a]amix=inputs=2:normalize=0:duration=longest[m];"
        f"[m]{LOUDNORM},aresample={rate}[out]"
    )
    mixed = _ffmpeg([vocal_wav, instrumental_wav], graph, ["-c:a", "pcm_s16le"], ".wav")
    return _apply(mixed, watermark)


def to_mp3(
    audio_wav: bytes,
    gain_db: float = 0.0,
    watermark: Callable[[bytes], bytes] | None = None,
) -> bytes:
    """The `speech` branch's output step: no mix, same level and same tagging."""
    gain = clamp_gain_db(gain_db)
    graph = f"[0:a]volume={gain:.2f}dB,{LOUDNORM}[out]"
    levelled = _ffmpeg([audio_wav], graph, ["-c:a", "pcm_s16le"], ".wav")
    return _apply(levelled, watermark)


def sum_stems(stems: list[bytes]) -> bytes:
    """Add N stems back into one wav, for models that split further than 2 ways.

    HTDemucs returns drums/bass/other where we want a single instrumental.
    `normalize=0` again: these stems sum back to the original mix by
    construction, and scaling them down would make the backing track quiet.
    """
    if not stems:
        raise MixError("nothing to sum")
    if len(stems) == 1:
        return stems[0]
    labels = "".join(f"[{i}:a]" for i in range(len(stems)))
    graph = f"{labels}amix=inputs={len(stems)}:normalize=0:duration=longest[out]"
    return _ffmpeg(stems, graph, ["-c:a", "pcm_s16le"], ".wav")
