"""AudioSeal watermarking of the finished output (plan §8, "Cân nhắc thêm").

The plan lists this as optional at MVP and names it as the thing to add before
the app is opened to the public, which is what this is. Meta's AudioSeal (MIT)
embeds an inaudible, 16-bit message into the audio that survives mp3 encoding,
so a file that comes back in a complaint can be checked against
`Watermarker.detect` instead of being argued about.

It pairs with the two safety features that already exist and covers what
neither can: the `AI-generated` comment tag in `mixing` is stripped by the
first re-encode anyone does, and `audit` proves what *we* ran but says nothing
about a file someone hands us. The watermark travels with the audio.

Four decisions worth keeping:

* **The model is 16kHz only.** AudioSeal 0.2 stopped resampling internally
  ("the user is responsible for providing the correct sample rate"), so the
  watermark is computed on a 16kHz mono view of the mix and then resampled back
  up and added to the full-rate, full-channel file. That is the pre-0.2
  behaviour, done here where it is visible: the music itself is never
  downsampled, it only gains a band-limited signal on top.

* **It runs after loudnorm and before the mp3 encode.** Both neighbours would
  damage it in the other order — `loudnorm` applies a gain and a limiter, which
  would rescale a watermark added before it. `mixing` takes this as a callable
  for exactly that reason.

* **Chunked, for the same reason conversion is.** One forward pass over an 8
  minute file allocates gigabytes in the SEANet stack; 30s windows keep it
  flat. The windows overlap and are joined with a *linear* fade, unlike the
  equal-power fade in `audio_utils`: there the two sides of a join are the same
  audio converted twice by a diffusion model and phase-incoherent, here they
  are two watermarks of the same audio and correlated, so equal-power would
  bump the join instead of smoothing it.

* **One fixed message, not a job id.** 16 bits cannot hold a 128-bit job id, so
  `WATERMARK_MESSAGE` identifies this deployment and nothing more: it separates
  "we made this" from "something else made this" and from a false positive on
  audio nobody watermarked. The audit trail is what maps a file to a job.

Failure is loud on purpose — see `enabled`.

Check a file after the fact (needs Modal credentials, no GPU):

    modal run -m modal_app.watermark --path suspect.mp3
"""

# No `from __future__ import annotations`: modal.parameter() reads the raw
# class annotation, and this module sits next to ones that use it.
import os

import modal
import numpy as np

from .app import MODEL_DIR, app, base_image, model_vol

# 0.2.0 is the release that removed internal resampling; pinned because that
# behaviour is the thing this module is written around.
AUDIOSEAL_SPEC = "audioseal==0.2.0"
GENERATOR_CARD = "audioseal_wm_16bits"
DETECTOR_CARD = "audioseal_detector_16bits"

# What the checkpoints were trained at. Not a preference.
MODEL_SAMPLE_RATE = 16000

# The 16 bits every output carries. A deployment id, not a job id: it answers
# "did this come from us", and `audit` answers "which job". Mixed bits rather
# than 0x0000 or 0xFFFF, which are the patterns an unwatermarked file is most
# likely to decode to by chance.
WATERMARK_MESSAGE = 0xC7A9
MESSAGE_BITS = 16

# Window and join, in seconds. The overlap matches `audio_utils.CHUNK_OVERLAP_SEC`
# because the reasoning about audible joins is the same, even though the fade is not.
WINDOW_SEC = 30.0
WINDOW_OVERLAP_SEC = 0.2

# `detect_watermark` returns the fraction of frames that look watermarked.
DETECTION_THRESHOLD = 0.5
# Out of 16. Below this the message is noise; a genuine watermark that has been
# through an mp3 encode still decodes most of its bits.
MESSAGE_MATCH_BITS = 14

# Set WATERMARK=0 to ship unwatermarked output. See `enabled`.
ENV_FLAG = "WATERMARK"
OFF_VALUES = {"0", "false", "no", "off"}

# Four cores for a couple of minutes costs less than a cent and roughly
# quarters the wall clock; the SEANet stack is small enough that a GPU would be
# mostly cold-start.
CPU_COUNT = 4

# The first variable audioseal's loader checks (`loader._get_cache_dir`).
# Pointing it at the model Volume is what makes the checkpoints download once
# rather than once per cold container; named here because a Modal Image does
# not expose its own build steps for a test to read back.
CACHE_ENV = "AUDIOSEAL_CACHE_DIR"

watermark_image = base_image.pip_install(AUDIOSEAL_SPEC).env({CACHE_ENV: MODEL_DIR})


class WatermarkError(RuntimeError):
    """The watermark could not be applied. Deliberately fatal to the job."""


def enabled(value: str | None = None) -> bool:
    """Whether output gets watermarked. On unless `WATERMARK` says otherwise.

    Default-on because a safety feature that defaults off is one nobody
    notices is broken, and because the plan puts this before going public. The
    escape hatch is an env var rather than a code change so that a bad HF
    download can be switched off without a deploy.

    There is deliberately no third state. If watermarking is on and it fails,
    the job fails: shipping audio that is silently *not* watermarked while the
    logs record that it was is worse than shipping nothing.
    """
    raw = os.environ.get(ENV_FLAG, "") if value is None else value
    # Unset reads as "" , which is not an off value, so it stays on.
    return raw.strip().lower() not in OFF_VALUES


def message_bits(value: int = WATERMARK_MESSAGE, nbits: int = MESSAGE_BITS) -> list[int]:
    """`WATERMARK_MESSAGE` as the bit list AudioSeal wants, most significant first."""
    if not 0 <= value < (1 << nbits):
        raise WatermarkError(f"message must fit in {nbits} bits, got {value}")
    return [(value >> shift) & 1 for shift in reversed(range(nbits))]


def bits_to_int(bits: list[int]) -> int:
    """The inverse of `message_bits`, for reporting what a detector decoded."""
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def window_bounds(total: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Cover `total` samples with overlapping windows of at most `window`.

    Sample counts rather than seconds so the arithmetic is exact and testable
    without any audio.
    """
    if total <= 0:
        raise WatermarkError("nothing to watermark")
    if window <= overlap or overlap < 0:
        raise WatermarkError(f"window {window} must be longer than overlap {overlap}")
    if total <= window:
        return [(0, total)]

    step = window - overlap
    bounds = []
    start = 0
    while start < total:
        end = min(start + window, total)
        bounds.append((start, end))
        if end >= total:
            break
        start += step
    return bounds


def blend(parts: list[np.ndarray], bounds: list[tuple[int, int]], total: int) -> np.ndarray:
    """Lay windowed watermarks back onto one timeline, fading across overlaps.

    Linear, not equal-power — see the module docstring. The two sides of a join
    are watermarks of the same audio, so they add rather than cancel and a
    linear fade holds the level flat.
    """
    if len(parts) != len(bounds):
        raise WatermarkError(f"got {len(parts)} windows for {len(bounds)} bounds")
    out = np.zeros(total, dtype=np.float32)
    filled = 0
    for (start, end), part in zip(bounds, parts, strict=True):
        part = np.asarray(part, dtype=np.float32)[: end - start]
        # A model may return a shorter tail than it was given; pad rather than
        # shift everything after it.
        if len(part) < end - start:
            part = np.pad(part, (0, end - start - len(part)))
        overlap = min(filled - start, len(part))
        if overlap > 0:
            fade = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
            out[start:filled] = out[start:filled] * (1.0 - fade) + part[:overlap] * fade
            out[filled:end] = part[overlap:]
        else:
            out[start:end] = part
        filled = max(filled, end)
    return out


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Mono float32 from one rate to another, through ffmpeg's resampler.

    A wav round trip rather than a Python resampler: `audio_utils` already owns
    ffmpeg decoding, and linear interpolation over a signal that has to survive
    an mp3 encode is not good enough.

    The signal is scaled to full range before the 16-bit round trip and back
    afterwards. Without that, a watermark that peaks around -45 dBFS would be
    quantised against a floor only ~45 dB below itself; the scaling is exactly
    linear, so it costs nothing and moves that floor out of the way.
    """
    from .audio_utils import decode_audio, encode_wav

    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max())
    if peak <= 0.0:
        return np.zeros(int(len(audio) * target_rate / source_rate), dtype=np.float32)
    scaled = np.asarray(audio, dtype=np.float32) / peak
    return decode_audio(encode_wav(scaled, source_rate), target_rate) * peak


def fit(signal: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to `length`. Resampling lands a sample or two off."""
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)))


@app.cls(
    image=watermark_image,
    cpu=CPU_COUNT,
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,
    timeout=900,
    max_containers=4,
)
class Watermarker:
    """Embed and detect. Both directions live here so they cannot drift apart."""

    @modal.enter()
    def load(self):
        import torch
        from audioseal import AudioSeal

        torch.set_num_threads(CPU_COUNT)
        self.generator = AudioSeal.load_generator(GENERATOR_CARD).eval()
        self.detector = AudioSeal.load_detector(DETECTOR_CARD).eval()
        self.message = torch.tensor([message_bits()], dtype=torch.long)
        # Downloaded on the first container only; every later one reads the
        # Volume.
        model_vol.commit()

    def _windows(self, samples: int) -> list[tuple[int, int]]:
        return window_bounds(
            samples,
            int(WINDOW_SEC * MODEL_SAMPLE_RATE),
            int(WINDOW_OVERLAP_SEC * MODEL_SAMPLE_RATE),
        )

    def _generate(self, mono: np.ndarray) -> np.ndarray:
        """The watermark signal for 16kHz mono audio, same length as its input."""
        import torch

        bounds = self._windows(len(mono))
        parts = []
        with torch.no_grad():
            for start, end in bounds:
                window = torch.from_numpy(np.ascontiguousarray(mono[start:end])).view(1, 1, -1)
                # `message=` on every call: left to itself the generator draws a
                # random message once per container, which would make two halves
                # of one deployment sign differently.
                parts.append(
                    self.generator.get_watermark(window, message=self.message)
                    .view(-1)
                    .cpu()
                    .numpy()
                )
        return blend(parts, bounds, len(mono))

    @modal.method()
    def embed(self, audio_wav: bytes) -> bytes:
        """Watermark a 16-bit PCM wav, keeping its sample rate and channels.

        Input is the mixed, loudness-normalised wav from `mixing`; output is the
        same file with the watermark added, ready to encode.
        """
        import time

        from .audio_utils import AudioError, decode_audio, decode_wav_channels, encode_wav_channels

        started = time.time()
        try:
            frames, sample_rate = decode_wav_channels(audio_wav)
            mono = decode_audio(audio_wav, MODEL_SAMPLE_RATE)
        except AudioError as exc:
            raise WatermarkError(f"unreadable audio: {exc}") from exc
        if not len(frames):
            raise WatermarkError("nothing to watermark")

        signal = resample(self._generate(mono), MODEL_SAMPLE_RATE, sample_rate)
        # The same mono watermark into every channel, which is what the mono
        # detector expects to find after it downmixes.
        marked = frames + fit(signal, len(frames))[:, None]

        print(
            f"[Watermarker] {len(frames) / sample_rate:.1f}s x {frames.shape[1]}ch "
            f"in {time.time() - started:.1f}s"
        )
        # `encode_wav_channels` clips, and it has ~1 dB of room to do it in:
        # `mixing` normalises to -1.0 dBTP before this runs.
        return encode_wav_channels(marked, sample_rate)

    @modal.method()
    def detect(self, audio: bytes) -> dict:
        """Is this ours? Takes any format ffmpeg reads, including a re-encode.

        `probability` is the share of frames the detector calls watermarked;
        `matching_bits` is how much of `WATERMARK_MESSAGE` survived. A file that
        was never watermarked scores low on the first and decodes a random
        message, which is why both are reported rather than just a verdict.
        """
        import torch

        from .audio_utils import decode_audio

        mono = decode_audio(audio, MODEL_SAMPLE_RATE)
        expected = message_bits()
        probabilities, votes, weights = [], [], []
        with torch.no_grad():
            for start, end in self._windows(len(mono)):
                window = torch.from_numpy(np.ascontiguousarray(mono[start:end])).view(1, 1, -1)
                probability, bits = self.detector.detect_watermark(window)
                probabilities.append(float(probability[0]))
                votes.append(bits[0].cpu().numpy())
                weights.append(end - start)

        weight = np.asarray(weights, dtype=np.float64)
        probability = float(np.average(probabilities, weights=weight))
        # Majority vote across windows: every window carries the same message,
        # so a window that sits under an instrumental break is outvoted.
        decoded = (np.average(np.stack(votes), axis=0, weights=weight) >= 0.5).astype(int)
        matching = int((decoded == np.asarray(expected)).sum())

        return {
            "watermarked": probability >= DETECTION_THRESHOLD,
            "ours": probability >= DETECTION_THRESHOLD and matching >= MESSAGE_MATCH_BITS,
            "probability": round(probability, 4),
            "matching_bits": matching,
            "message": bits_to_int(list(decoded)),
            "expected_message": WATERMARK_MESSAGE,
            "seconds": round(len(mono) / MODEL_SAMPLE_RATE, 1),
        }


@app.local_entrypoint()
def check(path: str) -> None:
    """Answer a complaint: did this file come out of this app?"""
    import json
    from pathlib import Path

    print(json.dumps(Watermarker().detect.remote(Path(path).read_bytes()), indent=2))
