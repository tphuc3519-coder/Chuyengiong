"""Audio helpers with no Modal, torch or GPU dependency.

Everything here is plain numpy plus an ffmpeg subprocess for decoding, so the
chunking and crossfade rules from the plan can be unit tested in CI without a
GPU. `conversion.py` is the only caller that matters; keep it that way, because
these functions are also the pieces Phase 3 will reuse for mixing input.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

# --- product limits -------------------------------------------------------
# Validated here rather than in the frontend: the browser is not the only
# caller and a bad reference is cheaper to reject than to convert badly.

MODES = ("speech", "singing")

REFERENCE_MIN_SEC = 5.0
# The plan says "cắt lấy 30s đầu", but seed-vc runs source and reference inside
# one 30s context window (`max_context_window` in inference.py), so a 30s
# reference would leave no room at all for source audio. 20s keeps a 10s source
# window per forward pass while staying well past the point where a longer
# reference stops improving timbre.
REFERENCE_MAX_SEC = 20.0

SOURCE_MAX_SEC = 15 * 60.0

DIFFUSION_STEPS_MIN = 10
DIFFUSION_STEPS_MAX = 100
DEFAULT_DIFFUSION_STEPS = {"speech": 25, "singing": 50}

# Large shifts push speech F0 outside its natural range and smear tone in
# tonal languages; singing carries the melody so it tolerates more.
MAX_SEMITONE_SHIFT = {"speech": 8, "singing": 12}

# Chunking defaults. `target`/`max`/`min` are the plan's; `search` is the
# half-width of the window we look inside for a quiet frame to cut at.
CHUNK_TARGET_SEC = 30.0
CHUNK_MAX_SEC = 40.0
CHUNK_MIN_SEC = 10.0
CHUNK_SEARCH_SEC = 8.0
CHUNK_OVERLAP_SEC = 0.2


class AudioError(ValueError):
    """Input the user can fix: wrong length, wrong format, undecodable."""


# --- wav bytes <-> array --------------------------------------------------


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 in [-1, 1] -> 16-bit PCM wav bytes."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise AudioError(f"expected mono audio, got shape {audio.shape}")
    samples = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM wav bytes -> (mono float32, sample_rate). No resampling."""
    try:
        with wave.open(io.BytesIO(data), "rb") as src:
            channels = src.getnchannels()
            sample_rate = src.getframerate()
            if src.getsampwidth() != 2:
                raise AudioError("only 16-bit PCM wav is supported here")
            raw = src.readframes(src.getnframes())
    except wave.Error as exc:  # not a wav at all
        raise AudioError(f"not a readable wav file: {exc}") from exc
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def encode_wav_channels(audio: np.ndarray, sample_rate: int) -> bytes:
    """Interleaved float32 `(frames, channels)` in [-1, 1] -> 16-bit PCM wav.

    The multi-channel counterpart of `encode_wav`, and the only reason it
    exists is watermarking: the final mix is stereo, and a step that has to
    read it, add something and write it back must not quietly fold it to mono
    on the way through.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise AudioError(f"expected (frames, channels), got shape {audio.shape}")
    samples = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(audio.shape[1])
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue()


def decode_wav_channels(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM wav -> (float32 `(frames, channels)`, sample_rate).

    `decode_wav` downmixes; this one keeps every channel, so a decode/encode
    round trip through it is lossless and leaves a stereo mix stereo.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as src:
            channels = src.getnchannels()
            sample_rate = src.getframerate()
            if src.getsampwidth() != 2:
                raise AudioError("only 16-bit PCM wav is supported here")
            raw = src.readframes(src.getnframes())
    except wave.Error as exc:
        raise AudioError(f"not a readable wav file: {exc}") from exc
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio.reshape(-1, channels), sample_rate


def decode_audio(data: bytes, sample_rate: int) -> np.ndarray:
    """Decode any ffmpeg-readable audio to mono float32 at `sample_rate`.

    Goes through a temp file rather than a pipe: mp4/m4a containers need to
    seek, and that is exactly the format an iPhone recording arrives in.
    """
    if not data:
        raise AudioError("empty audio file")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input"
        src.write_bytes(data)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "pipe:1",
            ],
            capture_output=True,
        )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(f"could not decode audio: {detail[-1] if detail else 'no output'}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    return len(audio) / float(sample_rate)


# --- validation -----------------------------------------------------------


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise AudioError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def prepare_reference(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate the voice sample and trim it to the usable window."""
    seconds = duration_sec(audio, sample_rate)
    if seconds < REFERENCE_MIN_SEC:
        raise AudioError(
            f"reference voice is {seconds:.1f}s, need at least {REFERENCE_MIN_SEC:.0f}s"
        )
    limit = int(REFERENCE_MAX_SEC * sample_rate)
    return audio[:limit] if len(audio) > limit else audio


def check_source(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    seconds = duration_sec(audio, sample_rate)
    if seconds <= 0:
        raise AudioError("source audio is empty")
    if seconds > SOURCE_MAX_SEC:
        raise AudioError(
            f"source is {seconds / 60:.1f} minutes, the limit is {SOURCE_MAX_SEC / 60:.0f} minutes"
        )
    return audio


def clamp_semitone_shift(shift: int, mode: str) -> int:
    limit = MAX_SEMITONE_SHIFT[check_mode(mode)]
    return int(max(-limit, min(limit, int(shift))))


def clamp_diffusion_steps(steps: int | None, mode: str) -> int:
    if not steps:
        return DEFAULT_DIFFUSION_STEPS[check_mode(mode)]
    return int(max(DIFFUSION_STEPS_MIN, min(DIFFUSION_STEPS_MAX, int(steps))))


# --- chunking -------------------------------------------------------------


def frame_rms(audio: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS per non-overlapping frame. Frame i covers [i*frame_len, (i+1)*frame_len)."""
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    frames = np.asarray(audio[: n_frames * frame_len], dtype=np.float64).reshape(
        n_frames, frame_len
    )
    return np.sqrt((frames**2).mean(axis=1))


def split_at_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sec: float = CHUNK_TARGET_SEC,
    max_sec: float = CHUNK_MAX_SEC,
    min_sec: float = CHUNK_MIN_SEC,
    search_sec: float = CHUNK_SEARCH_SEC,
    overlap_sec: float = CHUNK_OVERLAP_SEC,
) -> list[np.ndarray]:
    """Split into chunks that overlap by `overlap_sec`, cutting at quiet frames.

    The cut point for each boundary is the lowest-RMS frame inside
    [target - search, target + search], clamped to [min_sec, max_sec] from the
    previous cut — so a chunk is never longer than `max_sec` even when nothing
    in the window is actually silent. A trailing chunk shorter than `min_sec`
    is merged into the one before it, which is the only case where a chunk may
    reach `max_sec + min_sec`.

    Consecutive chunks share their last/first `overlap_sec` of audio; feed the
    converted chunks to `crossfade_concat` with the same overlap to get the
    original timeline back.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise AudioError("cannot split empty audio")
    if len(audio) <= int(max_sec * sample_rate):
        return [audio]

    frame_len = max(1, int(0.02 * sample_rate))  # 20 ms
    rms = frame_rms(audio, frame_len)

    cuts = [0]
    start = 0
    while len(audio) - start > int(max_sec * sample_rate):
        low = start + int(min_sec * sample_rate)
        high = start + int(max_sec * sample_rate)
        target = start + int(target_sec * sample_rate)
        low = max(low, target - int(search_sec * sample_rate))
        high = min(high, target + int(search_sec * sample_rate))
        low = min(low, high)

        first_frame = low // frame_len
        last_frame = min(high // frame_len, len(rms) - 1)
        if last_frame > first_frame:
            window = rms[first_frame : last_frame + 1]
            cut = int((first_frame + int(np.argmin(window))) * frame_len + frame_len // 2)
        else:
            cut = int(high)
        cuts.append(cut)
        start = cut

    # Tail too short to convert on its own: fold it into the previous chunk.
    if len(cuts) > 1 and len(audio) - cuts[-1] < int(min_sec * sample_rate):
        cuts.pop()

    overlap = int(overlap_sec * sample_rate)
    chunks = []
    for i, cut in enumerate(cuts):
        begin = cut if i == 0 else max(0, cut - overlap)
        end = cuts[i + 1] if i + 1 < len(cuts) else len(audio)
        chunks.append(audio[begin:end])
    return chunks


def crossfade_concat(
    chunks: list[np.ndarray], sample_rate: int, overlap_sec: float = CHUNK_OVERLAP_SEC
) -> np.ndarray:
    """Join overlapping chunks with an equal-power crossfade.

    Equal-power (`sqrt`) rather than linear: the two sides of a join are the
    same material converted twice, which diffusion makes phase-incoherent, and
    a linear fade dips ~3 dB in the middle of every join.
    """
    if not chunks:
        raise AudioError("nothing to join")
    out = np.asarray(chunks[0], dtype=np.float32).copy()
    for chunk in chunks[1:]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) == 0:
            continue
        overlap = min(int(overlap_sec * sample_rate), len(out), len(chunk))
        if overlap <= 0:
            out = np.concatenate([out, chunk])
            continue
        t = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        joined = out[-overlap:] * np.sqrt(1.0 - t) + chunk[:overlap] * np.sqrt(t)
        out = np.concatenate([out[:-overlap], joined, chunk[overlap:]])
    return out
