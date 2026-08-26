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
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

# -14 LUFS with 1 dB of headroom: the streaming reference, and the right target
# for something that will be listened to on a phone.
LOUDNORM = "loudnorm=I=-14:TP=-1.0"
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


def mix(vocal_wav: bytes, instrumental_wav: bytes, vocal_gain_db: float = 0.0) -> bytes:
    """Converted vocal over the original instrumental -> mp3 bytes.

    The two inputs are rarely the same length — conversion resamples and joins
    chunks — so `amix` runs to the longer of the two rather than truncating the
    song at whichever stem ends first.
    """
    gain = clamp_gain_db(vocal_gain_db)
    graph = (
        f"[0:a]volume={gain:.2f}dB[v];"
        f"[v][1:a]amix=inputs=2:normalize=0:duration=longest[m];"
        f"[m]{LOUDNORM}[out]"
    )
    return _ffmpeg([vocal_wav, instrumental_wav], graph, _mp3_args(), ".mp3")


def to_mp3(audio_wav: bytes, gain_db: float = 0.0) -> bytes:
    """The `speech` branch's output step: no mix, same level and same tagging."""
    gain = clamp_gain_db(gain_db)
    graph = f"[0:a]volume={gain:.2f}dB,{LOUDNORM}[out]"
    return _ffmpeg([audio_wav], graph, _mp3_args(), ".mp3")


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
