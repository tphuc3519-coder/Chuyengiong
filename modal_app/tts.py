"""Text to speech, as the third way of producing something to convert.

    text ──► Synthesizer (MMS-TTS) ──► spoken.wav ──► VoiceConverter ──► output

This module owns only the first arrow. Nothing here knows about the reference
voice: the timbre comes from the same Seed-VC pass the `speech` branch already
runs, so a voice sample that works there works here, and everything downstream —
pitch auto-detect, loudness, watermark, the consent gate — is the code that was
already shipping rather than a second copy of it.

Two consequences worth naming, because they are the reason it is built this way:

* **The synthetic voice is a stand-in, not the product.** MMS-TTS speaks in its
  own single speaker; the conversion is what makes it the user's voice. So the
  checkpoint is chosen for coverage and size, not for beauty.
* **The pitch shift is measured, not assumed.** `pipeline._resolve_shift` runs
  on `spoken.wav` against the reference exactly as it does for an uploaded
  recording — the MMS speaker for a language has one register and the target
  may be nowhere near it.

Why MMS-TTS (`facebook/mms-tts-<iso639-3>`) and not a zero-shot cloning TTS:
it is a plain `transformers` VITS checkpoint of ~145 MB that loads on CPU, it
covers Vietnamese properly, and cloning is already handled one step later —
a zero-shot TTS would duplicate that job with a second set of weights, a second
GPU, and a worse claim on languages.

**Numbers and symbols are not spoken.** MMS-TTS tokenises characters against a
per-language vocabulary that holds letters and punctuation; a digit is not in
it and is dropped without a word of complaint. "25 tuổi" is read as "tuổi".
Writing them out is the fix, and the UI says so — expanding them here would
mean per-language number grammar, which is a bigger and more wrongable job than
it looks (`hai mươi lăm`, not `hai năm`).

Smoke test (needs Modal credentials, no GPU):

    modal run -m modal_app.tts --text "Xin chào, đây là giọng của tôi."
"""

# NB: no `from __future__ import annotations` — modal.parameter() reads the raw
# class annotation and cannot resolve a stringified one.
import re

import modal

from .app import MODEL_DIR, app, base_image, model_vol

# Latin-script languages whose MMS checkpoint tokenises the text as written.
#
# MMS covers ~1100 languages, but the ones outside a Latin script need their
# input romanised with `uroman` first, and a checkpoint fed unromanised text
# returns silence rather than an error. Only languages that need no such step
# are listed, and `Synthesizer.load` refuses anything whose tokenizer disagrees
# — so adding one is a matter of putting it here and running the smoke test,
# not of hoping.
LANGUAGES = {
    "vie": "Tiếng Việt",
    "eng": "English",
    "ind": "Bahasa Indonesia",
    "fra": "Français",
    "spa": "Español",
    "deu": "Deutsch",
    "por": "Português",
    "ita": "Italiano",
}
DEFAULT_LANGUAGE = "vie"

# ~2000 characters is 2 to 3 minutes of speech, which is where the Seed-VC pass
# after it starts to be the expensive half of a job rather than a step in one.
MAX_TEXT_CHARS = 2000
# One synthesis call per segment. VITS predicts a duration per token and its
# error accumulates over a long stretch, so a paragraph handed over whole comes
# back with its timing drifting; a sentence at a time does not.
SEGMENT_MAX_CHARS = 200
# Silence inserted between segments, in place of the pause the split removed.
SEGMENT_GAP_SEC = 0.25
# Room for the converter to work with at both ends. Seed-VC's first and last
# frames are its least certain, and having them land on silence rather than on
# the first syllable is free.
EDGE_PAD_SEC = 0.15

SPEAKING_RATE_MIN = 0.5
SPEAKING_RATE_MAX = 2.0
DEFAULT_SPEAKING_RATE = 1.0

# Peak the synthesised wav is normalised to before it leaves this container.
# VITS output level varies with the sentence; `loudnorm` at the end of the
# pipeline sets the real level, and this only keeps the converter's input in a
# consistent place.
TARGET_PEAK = 0.9

# Sentence enders, plus any line break. Vietnamese uses the same set.
_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+|\n+")


class TtsError(ValueError):
    """Text the user can fix: empty, too long, or in no language we speak."""


def check_language(language: str) -> str:
    if language not in LANGUAGES:
        raise TtsError(f"language must be one of {tuple(LANGUAGES)}, got {language!r}")
    return language


def model_id(language: str) -> str:
    return f"facebook/mms-tts-{check_language(language)}"


def clamp_speaking_rate(rate: float | None) -> float:
    """Out of range is clamped, not refused — a slider is a client, not a user."""
    try:
        value = float(rate) if rate else DEFAULT_SPEAKING_RATE
    except (TypeError, ValueError):
        return DEFAULT_SPEAKING_RATE
    return max(SPEAKING_RATE_MIN, min(SPEAKING_RATE_MAX, value))


def _wrap(piece: str, max_chars: int) -> list[str]:
    """One over-long sentence, broken at the latest clause boundary that fits."""
    parts: list[str] = []
    while len(piece) > max_chars:
        window = piece[:max_chars]
        # A clause boundary first, and keep the punctuation with the clause it
        # closes: the model reads a comma as the pause it is.
        cut = max(window.rfind(", "), window.rfind("; "))
        cut = cut + 1 if cut > 0 else window.rfind(" ")
        # A single unbroken run this long is not language; cut it where it fits
        # rather than hand the model a paragraph.
        if cut <= 0:
            cut = max_chars
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


def split_text(text: str, max_chars: int = SEGMENT_MAX_CHARS) -> list[str]:
    """Text as a list of segments to synthesise one at a time.

    Sentence boundaries first, because that is where a pause belongs anyway;
    anything still too long is broken at a clause. Empty pieces are dropped, so
    blank lines and doubled punctuation cost nothing.
    """
    segments: list[str] = []
    for piece in _SENTENCE_END.split(text.strip()):
        piece = piece.strip()
        if piece:
            segments.extend(_wrap(piece, max_chars))
    return segments


def check_text(text: str) -> str:
    """Validate what `/submit` was given. Returns the text to store.

    The "no letters" case is the one worth having: MMS drops every character it
    has no token for, so a line of digits synthesises to silence and would
    otherwise reach the user as a job that succeeded and produced nothing.
    """
    if not isinstance(text, str):
        raise TtsError("text must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise TtsError("no text to read: write something first")
    if len(cleaned) > MAX_TEXT_CHARS:
        raise TtsError(f"text is {len(cleaned)} characters, the limit is {MAX_TEXT_CHARS}")
    if not any(ch.isalpha() for ch in cleaned):
        raise TtsError("no words to read: numbers and symbols are not spoken, write them out")
    if not split_text(cleaned):
        raise TtsError("no text to read: write something first")
    return cleaned


# transformers is the whole dependency: MMS-TTS is a VITS checkpoint, and
# `VitsModel` synthesises straight to a waveform with no vocoder to install and
# no phonemizer — the tokenizer works on characters. The pin matches the one
# `conversion.py` already builds against.
tts_image = base_image.pip_install("transformers==4.46.3").env({"HF_HOME": MODEL_DIR})


@app.cls(
    image=tts_image,
    # No GPU on purpose. VITS is ~145 MB and synthesises a sentence in well
    # under real time on a few cores, so a GPU would spend most of a job cold
    # starting; the GPU minutes in this pipeline belong to the conversion.
    cpu=4,
    memory=4096,
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,
    timeout=900,
    max_containers=4,
)
class Synthesizer:
    language: str = modal.parameter(default=DEFAULT_LANGUAGE)

    @modal.enter()
    def load(self) -> None:
        """Runs once per container. Weights land on the Volume via HF_HOME."""
        import os

        from transformers import AutoTokenizer, VitsModel

        repo = model_id(self.language)
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        # The romanisation gate, checked rather than assumed: a checkpoint that
        # wants `uroman` input returns silence for text it cannot read, and a
        # silent success is the worst failure this pipeline can produce.
        if getattr(self.tokenizer, "is_uroman", False):
            raise TtsError(f"{repo} needs romanised input; {self.language} is not supported")

        self.model = VitsModel.from_pretrained(repo)
        self.model.eval()
        self.sr = self.model.config.sampling_rate
        model_vol.commit()
        print(f"[Synthesizer] language={self.language} sr={self.sr}")

    @modal.method()
    def synthesize(self, text: str, speaking_rate: float = DEFAULT_SPEAKING_RATE) -> bytes:
        """Text in, 16-bit PCM wav out, at the checkpoint's own sample rate.

        The rate is not resampled here: `VoiceConverter.convert` decodes through
        ffmpeg at whatever rate its own checkpoint wants, so converting twice
        would only cost a generation of quality.
        """
        import time

        import numpy as np
        import torch

        from .audio_utils import encode_wav

        segments = split_text(check_text(text))
        rate = clamp_speaking_rate(speaking_rate)
        # `speaking_rate` scales the predicted durations, so it is read at
        # forward time and can be set per request without reloading anything.
        self.model.speaking_rate = rate

        started = time.time()
        gap = np.zeros(int(SEGMENT_GAP_SEC * self.sr), dtype=np.float32)
        pieces: list[np.ndarray] = []
        with torch.no_grad():
            for i, segment in enumerate(segments, start=1):
                inputs = self.tokenizer(segment, return_tensors="pt")
                # Every character was outside the vocabulary — a segment of
                # digits, or of a script this checkpoint does not read. Skip it
                # rather than ask the model to generate from nothing.
                if inputs["input_ids"].shape[-1] == 0:
                    print(f"[Synthesizer] segment {i}/{len(segments)} has no readable characters")
                    continue
                wave = self.model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
                if pieces:
                    pieces.append(gap)
                pieces.append(wave)

        if not pieces:
            raise TtsError("nothing in this text could be read out loud")

        pad = np.zeros(int(EDGE_PAD_SEC * self.sr), dtype=np.float32)
        audio = np.concatenate([pad, *pieces, pad])
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio * (TARGET_PEAK / peak)

        print(
            f"[Synthesizer] {len(segments)} segment(s) -> {len(audio) / self.sr:.1f}s "
            f"in {time.time() - started:.1f}s (rate={rate:.2f})"
        )
        return encode_wav(audio, self.sr)


# `speak`, not `main`: local entrypoint names are one flat namespace across the
# App, and `conversion.main` is already in it.
@app.local_entrypoint()
def speak(
    text: str,
    output: str = "spoken.wav",
    language: str = DEFAULT_LANGUAGE,
    speaking_rate: float = DEFAULT_SPEAKING_RATE,
) -> None:
    """Standalone smoke test: text in, one wav out. No reference, no GPU."""
    from pathlib import Path

    result = Synthesizer(language=language).synthesize.remote(
        text=text, speaking_rate=speaking_rate
    )
    Path(output).write_bytes(result)
    print(f"wrote {output} ({len(result) / 1e6:.1f} MB)")
