"""Text to speech, as the third way of producing something to convert.

    text ──► Synthesizer (MMS-TTS)  ──► spoken.wav ──► VoiceConverter ──► output
             KokoroSynthesizer (ja)

This module owns only the first arrow. Nothing here knows about the reference
voice: the timbre comes from the same Seed-VC pass the `speech` branch already
runs, so a voice sample that works there works here, and everything downstream —
pitch auto-detect, loudness, watermark, the consent gate — is the code that was
already shipping rather than a second copy of it. `synthesize()` is the seam:
callers name a language, not an engine.

Two consequences worth naming, because they are the reason it is built this way:

* **The synthetic voice is a stand-in, not the product.** Both engines speak in
  a single fixed speaker; the conversion is what makes it the user's voice. So
  a checkpoint is chosen for coverage, correctness and size, not for beauty.
* **The pitch shift is measured, not assumed.** `pipeline._resolve_shift` runs
  on `spoken.wav` against the reference exactly as it does for an uploaded
  recording — the synthetic speaker for a language has one register and the
  target may be nowhere near it.

Why MMS-TTS (`facebook/mms-tts-<iso639-3>`) and not a zero-shot cloning TTS:
it is a plain `transformers` VITS checkpoint of ~145 MB that loads on CPU, it
covers Vietnamese properly, and cloning is already handled one step later —
a zero-shot TTS would duplicate that job with a second set of weights, a second
GPU, and a worse claim on languages.

**Japanese does not go through MMS, and this is measured rather than assumed.**
MMS reads a non-Latin script by romanising it with `uroman` first, both when it
was trained and at inference. Run `uroman` on Japanese and the kanji come back
in *Mandarin*:

    今日はいい天気ですね。  ->  jinrihaiitianqidesune.
    私の名前は田中です。    ->  sinomingqianhatianzhongdesu.

(`kyou wa ii tenki desu ne` / `watashi no namae wa tanaka desu`; passing
`lcode='jpn'` changes nothing.) Kana survives, kanji does not — and Japanese
prose is roughly half kanji. That is the training text `facebook/mms-tts-jpn`
learned from, so neither correct romaji nor uroman's own output gets real
Japanese out of it. Hence a second engine for it, and only for it: Kokoro,
whose Japanese front end (`misaki[ja]`) is a dictionary-and-morphology G2P
that reads 今日 as `kʲoː` and 田中 as `tanaka`.

One engine would have been better. Two is what the language needs.

**Neither of them is read flat.** Both speak in one fixed speaker with no
emotion conditioning, so what they are given instead is a plan: `prosody.py`
turns the text into a `Beat` per sentence — pace, height, level, the length of
the silence after it, and a rise at the end of a question — and this module
spends it on whichever knobs its engine has. That plan is the difference
between a page read out and a page got through, and it is the only part of the
reading either engine could not have worked out from the sentence in front of
it: it is handed one at a time and cannot see the paragraph.

**Numbers and symbols are not spoken.** MMS-TTS tokenises characters against a
per-language vocabulary that holds letters and punctuation; a digit is not in
it and is dropped without a word of complaint. "25 tuổi" is read as "tuổi".
Writing them out is the fix, and the UI says so — expanding them here would
mean per-language number grammar, which is a bigger and more wrongable job than
it looks (`hai mươi lăm`, not `hai năm`).

Smoke test (needs Modal credentials, no GPU):

    modal run -m modal_app.tts --text "Xin chào, đây là giọng của tôi."
    modal run -m modal_app.tts --language jpn --text "今日はいい天気ですね。"
"""

# NB: no `from __future__ import annotations` — modal.parameter() reads the raw
# class annotation and cannot resolve a stringified one.
import re
from dataclasses import dataclass

import modal

from . import prosody
from .app import MODEL_DIR, app, base_image, model_vol

# ~2000 characters is 2 to 3 minutes of speech, which is where the Seed-VC pass
# after it starts to be the expensive half of a job rather than a step in one.
MAX_TEXT_CHARS = 2000
# One synthesis call per segment. VITS predicts a duration per token and its
# error accumulates over a long stretch, so a paragraph handed over whole comes
# back with its timing drifting; a sentence at a time does not.
SEGMENT_MAX_CHARS = 200

MMS = "mms"
KOKORO = "kokoro"


@dataclass(frozen=True)
class Language:
    """One language, and everything that differs about reading it out loud.

    `max_chars` and `segment_max_chars` are per language because a character is
    not a unit of speech: 2000 characters of Vietnamese is two or three
    minutes, and 2000 characters of Japanese — which writes a whole word in the
    space of one or two — is well over ten. The limits exist to bound how long
    the recording is, so they are set in the units each script actually spends.
    """

    label: str
    engine: str = MMS
    max_chars: int = MAX_TEXT_CHARS
    segment_max_chars: int = SEGMENT_MAX_CHARS
    # Whether Latin letters in this language are romaji to be read as its own
    # script rather than as themselves. See `to_kana`.
    romaji_input: bool = False
    # Kokoro only: its own one-letter language code, and which of its voices
    # reads. Which voice barely matters — Seed-VC replaces the timbre a step
    # later — so this is simply a clear, natural speaker to convert from.
    kokoro_code: str = ""
    voice: str = ""


# What we will read, and with what.
#
# The MMS entries are the Latin-script languages whose checkpoint tokenises the
# text as written. MMS covers ~1100 languages, but everything outside a Latin
# script has to be romanised with `uroman` first, and a checkpoint fed text it
# cannot read returns silence rather than an error — so only languages needing
# no such step are listed here, and `Synthesizer.load` refuses anything whose
# tokenizer disagrees. Adding one is a matter of putting it here and running the
# smoke test, not of hoping.
#
# Japanese is the exception that proves why: uroman renders its kanji in
# Mandarin (see the module docstring), so it reads through Kokoro instead.
JAPANESE = "jpn"
LANGUAGES = {
    "vie": Language("Tiếng Việt"),
    "eng": Language("English"),
    "ind": Language("Bahasa Indonesia"),
    "fra": Language("Français"),
    "spa": Language("Español"),
    "deu": Language("Deutsch"),
    "por": Language("Português"),
    "ita": Language("Italiano"),
    JAPANESE: Language(
        "日本語",
        engine=KOKORO,
        # ~700 Japanese characters is the same two to three minutes of speech
        # the Latin limit buys, and 80 per segment keeps the phoneme string
        # well under the 510 Kokoro truncates at.
        max_chars=700,
        segment_max_chars=80,
        romaji_input=True,
        kokoro_code="j",
        voice="jf_alpha",
    ),
}
DEFAULT_LANGUAGE = "vie"
# Silence between segments when nobody said how long it should be. `prosody`
# decides that per segment from the punctuation that closed it — a comma is not
# a full stop and neither is a blank line — so this is only the fallback for a
# caller that hands `_join` audio with no plan attached.
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

# Sentence enders, plus any line break. Vietnamese uses the same set as English.
#
# Two alternatives rather than one, because the trailing space is not optional
# in the same way for both. A Latin sentence ends "like this. And so" — the
# space is what tells a full stop from a decimal point. Japanese ends 「です。」
# and simply starts the next one, no space anywhere in the line, so its
# punctuation has to split on a zero-width match or a paragraph of it stays a
# single 700 character "sentence".
_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+|(?<=[。！？])\s*|\n+")
# Clause boundaries to fall back on inside an over-long sentence. The
# ideographic comma is here for the same reason as the ideographic full stop.
_CLAUSE_BREAKS = (", ", "; ", "、", "，")


class TtsError(ValueError):
    """Text the user can fix: empty, too long, or in no language we speak."""


def check_language(language: str) -> str:
    if language not in LANGUAGES:
        raise TtsError(f"language must be one of {tuple(LANGUAGES)}, got {language!r}")
    return language


def spec_for(language: str) -> Language:
    return LANGUAGES[check_language(language)]


def model_id(language: str) -> str:
    """The MMS checkpoint for `language`. Raises for a language MMS cannot read."""
    spec = spec_for(language)
    if spec.engine != MMS:
        raise TtsError(f"{language} does not read through MMS, it reads through {spec.engine}")
    return f"facebook/mms-tts-{language}"


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
        cut = max(window.rfind(mark) for mark in _CLAUSE_BREAKS)
        cut = cut + 1 if cut > 0 else window.rfind(" ")
        # No clause and no space — Japanese writes whole sentences without one,
        # and a Latin run this long is not language either. Cut where it fits.
        if cut <= 0:
            cut = max_chars
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


# A blank line: where one paragraph ends and the next begins. `_SENTENCE_END`
# splits on any run of newlines and so cannot tell that from a wrapped line, and
# the difference is three quarters of a second of silence and a pitch reset —
# see `prosody`.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


def split_blocks(text: str, max_chars: int = SEGMENT_MAX_CHARS) -> list[list[str]]:
    """Text as paragraphs of segments, which is what `prosody.plan` reads.

    Sentence boundaries first, because that is where a pause belongs anyway;
    anything still too long is broken at a clause. Empty pieces are dropped, so
    blank lines and doubled punctuation cost nothing — but *where* the blank
    lines were is kept, because two of the reading rules are about the start and
    the end of a paragraph.
    """
    blocks: list[list[str]] = []
    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        segments: list[str] = []
        for piece in _SENTENCE_END.split(paragraph.strip()):
            piece = piece.strip()
            if piece:
                segments.extend(_wrap(piece, max_chars))
        if segments:
            blocks.append(segments)
    return blocks


def split_text(text: str, max_chars: int = SEGMENT_MAX_CHARS) -> list[str]:
    """`split_blocks` with the paragraphs flattened away.

    Every caller that only needs to know how many times the model will be
    called, and `check_text`, which only needs to know whether the answer is
    zero.
    """
    return [segment for block in split_blocks(text, max_chars) for segment in block]


# Wapuro romaji writes ん before a vowel as "nn" or "n'", and `jaconv` reads
# only the apostrophe: "konnichiwa" comes back こんいちわ, a mora short and a
# different word. Rewriting the doubled n into the form it does read costs one
# regex — "onnanoko" and "sennin" come out おんなのこ and せんにん.
_ROMAJI_DOUBLE_N = re.compile(r"n{2,}(?=[aiueo])", re.IGNORECASE)


def to_kana(text: str) -> str:
    """Romaji in `text` as kana, leaving kana, kanji and punctuation alone.

    Japanese typed without an IME is romaji, and Kokoro's Japanese front end
    hands Latin letters straight through untouched — `konnichiwa` reaches the
    model as eleven Latin characters, which is not a reading of anything.

    Romaji is phonetic, so this is a spelling change rather than a translation
    and the result is the same audio: `kyou wa ii tenki desu ne.` comes out of
    the G2P as `kʲoː βa iː teŋkʲi desɨ ne.`, which is exactly what
    `今日はいい天気ですね。` gives. A Latin word inside Japanese text is read the
    same way — `私はTanakaです` becomes 私はたなかです — which is the right answer
    for a name and the closest available one for anything else.
    """
    import jaconv

    return jaconv.alphabet2kana(_ROMAJI_DOUBLE_N.sub("n'n", text))


def check_text(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Validate what `/submit` was given. Returns the text to store.

    The limit comes from the language because a character is not a unit of
    speech — see `Language`.

    The "no letters" case is the one worth having: MMS drops every character it
    has no token for, so a line of digits synthesises to silence and would
    otherwise reach the user as a job that succeeded and produced nothing.
    Kana and kanji are letters to `str.isalpha`, so Japanese passes it as
    written.
    """
    limit = spec_for(language).max_chars
    if not isinstance(text, str):
        raise TtsError("text must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise TtsError("no text to read: write something first")
    if len(cleaned) > limit:
        raise TtsError(f"text is {len(cleaned)} characters, the limit is {limit}")
    if not any(ch.isalpha() for ch in cleaned):
        raise TtsError("no words to read: numbers and symbols are not spoken, write them out")
    if not split_text(cleaned):
        raise TtsError("no text to read: write something first")
    return cleaned


def _join(segments, sample_rate: int, pauses: list[float] | None = None) -> bytes:
    """One wav out of the per-segment audio, with the pauses put back.

    `pauses[i]` is the silence that follows segment `i`, which `prosody.plan`
    read off the punctuation that closed it; the last entry is unused, because
    what follows the last segment is the end. Without a plan every gap is
    `SEGMENT_GAP_SEC`, which is what this did before there was one.

    Then a little silence at each end so the converter's least certain frames
    land on nothing rather than on the first syllable, and a fixed peak so both
    engines hand Seed-VC input at the same level. `loudnorm` sets the real level
    later — and it is why the per-segment gains are safe here: what this
    normalises is the loudest moment, so a louder exclamation raises that one
    sentence against the rest rather than the whole file against nothing.
    """
    import numpy as np

    from .audio_utils import encode_wav

    if not segments:
        raise TtsError("nothing in this text could be read out loud")

    gaps = list(pauses or [])
    gaps += [SEGMENT_GAP_SEC] * (len(segments) - len(gaps))
    pad = np.zeros(int(EDGE_PAD_SEC * sample_rate), dtype=np.float32)
    pieces = [pad]
    for i, audio in enumerate(segments):
        if i:
            gap = max(0.0, float(gaps[i - 1]))
            if gap > 0:
                pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
        pieces.append(np.asarray(audio, dtype=np.float32))
    pieces.append(pad)

    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio * (TARGET_PEAK / peak)
    return encode_wav(audio, sample_rate)


# Pinned, and pinned in *both* images below rather than left to the resolver.
#
# `base_image` carries torch 2.4.0 and no transformers, so anything that asks
# for transformers without a version gets the newest — today 5.x, which
# requires torch >= 2.5. It does not fail on that: it prints
#
#     [transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.0
#
# to the build log and carries on without torch, and every model class it
# exports becomes a stub that raises "requires the PyTorch library" the first
# time something builds one. For the Kokoro container that is inside
# `@modal.enter()`, on the first Japanese request, long after the deploy went
# green. 4.46.3 is the version `conversion.py` already builds against.
TRANSFORMERS_SPEC = "transformers==4.46.3"

# transformers is the whole dependency: MMS-TTS is a VITS checkpoint, and
# `VitsModel` synthesises straight to a waveform with no vocoder to install and
# no phonemizer — the tokenizer works on characters.
tts_image = base_image.pip_install(TRANSFORMERS_SPEC).env({"HF_HOME": MODEL_DIR})


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

        # Raises for a language that does not read through MMS at all, which is
        # a deployment mistake rather than something a request can cause.
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
        # VITS' two expressiveness knobs, as this checkpoint ships them.
        # `noise_scale` is how much the prior is allowed to vary — the pitch and
        # energy of the reading — and `noise_scale_duration` is the same for the
        # stochastic duration predictor, which is its rhythm. A style moves them
        # by a multiplier rather than setting a number, because the defaults are
        # per checkpoint and overwriting them with a constant would be a change
        # no style asked for.
        self.base_noise_scale = float(self.model.noise_scale)
        self.base_noise_scale_duration = float(self.model.noise_scale_duration)
        model_vol.commit()
        print(f"[Synthesizer] language={self.language} sr={self.sr}")

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaking_rate: float = DEFAULT_SPEAKING_RATE,
        emotion: str = prosody.DEFAULT_EMOTION,
        expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
    ) -> bytes:
        """Text in, 16-bit PCM wav out, at the checkpoint's own sample rate.

        One forward pass per sentence, and every one of them set up separately:
        `prosody.plan` says how fast this sentence goes, how much the duration
        predictor and the prior are allowed to wander, and what happens to the
        audio afterwards. All three are read at forward time, so a paragraph
        costs exactly what it did when every sentence was identical.

        The rate is not resampled here: `VoiceConverter.convert` decodes through
        ffmpeg at whatever rate its own checkpoint wants, so converting twice
        would only cost a generation of quality.
        """
        import time

        import numpy as np
        import torch

        spec = spec_for(self.language)
        blocks = split_blocks(check_text(text, self.language), spec.segment_max_chars)
        rate = clamp_speaking_rate(speaking_rate)
        beats = prosody.plan(
            blocks,
            emotion=emotion,
            speaking_rate=rate,
            expressiveness=expressiveness,
        )

        started = time.time()
        spoken: list[np.ndarray] = []
        pauses: list[float] = []
        with torch.no_grad():
            for i, beat in enumerate(beats, start=1):
                # Per sentence, not per container: all three are plain
                # attributes the forward pass reads, so nothing is reloaded.
                self.model.speaking_rate = clamp_speaking_rate(beat.rate)
                self.model.noise_scale = self.base_noise_scale * beat.variation
                self.model.noise_scale_duration = (
                    self.base_noise_scale_duration * beat.duration_variation
                )
                inputs = self.tokenizer(beat.text, return_tensors="pt")
                # Every character was outside the vocabulary — a segment of
                # digits, or of a script this checkpoint does not read. Skip it
                # rather than ask the model to generate from nothing. Its pause
                # goes with it, so the silence around the gap stays one pause.
                if inputs["input_ids"].shape[-1] == 0:
                    print(f"[Synthesizer] segment {i}/{len(beats)} has no readable characters")
                    continue
                audio = self.model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
                spoken.append(prosody.shape(audio, self.sr, beat))
                pauses.append(beat.pause_sec)

        wav = _join(spoken, self.sr, pauses)
        print(
            f"[Synthesizer] {len(beats)} segment(s) in {time.time() - started:.1f}s "
            f"(rate={rate:.2f} emotion={prosody.clean_emotion(emotion)} "
            f"expressiveness={prosody.clamp_expressiveness(expressiveness):.2f})"
        )
        return wav


# Kokoro is one pip package on top of the same base: an 82M StyleTTS2-style
# model plus `misaki`, its front end. Only the Japanese G2P is used here, and
# it needs no `espeak-ng` — that is the fallback for Kokoro's *European*
# languages, which read through MMS in this app.
#
# `TRANSFORMERS_SPEC` is in the same call rather than left to `kokoro`, which
# asks for transformers unversioned — see that constant for what the newest one
# does to a torch 2.4.0 image. Everything else resolves around the pins already
# in `base_image`: spacy 3.8 and thinc 8.3 come in and leave numpy at 1.26.4,
# so nothing here drags the audio stack onto numpy 2, and torch stays put.
kokoro_image = (
    base_image.pip_install(
        "kokoro==0.9.4",
        "misaki[ja]==0.9.4",
        "unidic-lite==1.0.8",
        TRANSFORMERS_SPEC,
    )
    # `misaki[ja]` depends on `unidic`, which ships no dictionary — it is a
    # downloader for one, and `fugashi` picks it over `unidic-lite` whenever
    # both are installed. The container then dies inside `Tagger()` with
    #
    #     param.cpp(69) [ifs] no such file or directory: .../unidic/dicdir/mecabrc
    #
    # before a word is read. Removing it leaves the dictionary that is actually
    # present. `python -m unidic download` is the other fix and costs ~700 MB
    # of image for readings `unidic-lite` already has.
    .run_commands("pip uninstall -y unidic")
    .env({"HF_HOME": MODEL_DIR})
)

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
# Kokoro truncates a phoneme string past this, with a warning and no error.
# `segment_max_chars` is set well under it; this is here to explain the margin.
KOKORO_MAX_PHONEMES = 510


@app.cls(
    image=kokoro_image,
    # CPU for the same reason as the MMS synthesiser: 82M parameters, and the
    # GPU minutes in this pipeline belong to the conversion.
    cpu=4,
    memory=4096,
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,
    timeout=900,
    max_containers=4,
)
class KokoroSynthesizer:
    """The engine for languages MMS cannot read as written. Today: Japanese."""

    language: str = modal.parameter(default=JAPANESE)

    @modal.enter()
    def load(self) -> None:
        import os

        from kokoro import KPipeline

        spec = spec_for(self.language)
        if spec.engine != KOKORO:
            raise TtsError(f"{self.language} does not read through Kokoro")
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.spec = spec
        # Weights, config and the voice pack all arrive through
        # `huggingface_hub`, so HF_HOME on the Volume is what keeps a second
        # container from downloading them again.
        self.pipeline = KPipeline(lang_code=spec.kokoro_code, repo_id=KOKORO_REPO)
        model_vol.commit()
        print(f"[KokoroSynthesizer] language={self.language} voice={spec.voice}")

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaking_rate: float = DEFAULT_SPEAKING_RATE,
        emotion: str = prosody.DEFAULT_EMOTION,
        expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
    ) -> bytes:
        """Text in, 16-bit PCM wav out at Kokoro's 24 kHz. Same contract as MMS.

        The same plan, with one knob fewer to spend it on: Kokoro takes a speed
        and nothing else, so the pace is set per sentence here and the pitch,
        the level and the final rise are all `prosody.shape`'s to apply. That is
        the same division as MMS — `noise_scale` is the only thing MMS adds —
        and the reading comes out the same shape either way.
        """
        import time

        import numpy as np

        spoken_text = check_text(text, self.language)
        # Before the split, not after: kana is what the segment budget is
        # measured in, and romaji is about twice as long as the kana it spells.
        if self.spec.romaji_input:
            spoken_text = to_kana(spoken_text)
        blocks = split_blocks(spoken_text, self.spec.segment_max_chars)
        rate = clamp_speaking_rate(speaking_rate)
        beats = prosody.plan(
            blocks,
            emotion=emotion,
            speaking_rate=rate,
            expressiveness=expressiveness,
        )

        started = time.time()
        spoken: list[np.ndarray] = []
        pauses: list[float] = []
        for i, beat in enumerate(beats, start=1):
            # `split_pattern=None`: the segments are already the split, and
            # letting Kokoro re-split on newlines it will not find only makes
            # the two disagree about where the pauses are.
            parts = [
                result.audio.cpu().numpy().astype(np.float32)
                for result in self.pipeline(
                    beat.text,
                    voice=self.spec.voice,
                    speed=clamp_speaking_rate(beat.rate),
                    split_pattern=None,
                )
                if result.audio is not None
            ]
            if not parts:
                print(f"[KokoroSynthesizer] segment {i}/{len(beats)} produced no audio")
                continue
            audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
            spoken.append(prosody.shape(audio, KOKORO_SAMPLE_RATE, beat))
            pauses.append(beat.pause_sec)

        wav = _join(spoken, KOKORO_SAMPLE_RATE, pauses)
        print(
            f"[KokoroSynthesizer] {len(beats)} segment(s) in {time.time() - started:.1f}s "
            f"(rate={rate:.2f} emotion={prosody.clean_emotion(emotion)} "
            f"expressiveness={prosody.clamp_expressiveness(expressiveness):.2f})"
        )
        return wav


def synthesize(
    language: str,
    text: str,
    speaking_rate: float,
    emotion: str = prosody.DEFAULT_EMOTION,
    expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
) -> bytes:
    """Read `text` with whichever engine speaks `language`, on its own container.

    The one place the engine split is resolved. `pipeline.py` asks for a
    language and gets audio back; which model produced it is this module's
    business, and adding a third engine should not reach the pipeline at all.
    The same is now true of how it is read: both engines take the same style,
    because the style is `prosody`'s and not either model's.
    """
    spec = spec_for(language)
    engine = KokoroSynthesizer if spec.engine == KOKORO else Synthesizer
    return engine(language=language).synthesize.remote(
        text=text,
        speaking_rate=speaking_rate,
        emotion=emotion,
        expressiveness=expressiveness,
    )


# `speak`, not `main`: local entrypoint names are one flat namespace across the
# App, and `conversion.main` is already in it.
@app.local_entrypoint()
def speak(
    text: str,
    output: str = "spoken.wav",
    language: str = DEFAULT_LANGUAGE,
    speaking_rate: float = DEFAULT_SPEAKING_RATE,
    emotion: str = prosody.DEFAULT_EMOTION,
    expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
) -> None:
    """Standalone smoke test: text in, one wav out. No reference, no GPU.

    modal run -m modal_app.tts --text "Xin chào."
    modal run -m modal_app.tts --language jpn --text "今日はいい天気ですね。"
    modal run -m modal_app.tts --emotion cheerful \
        --text "Chào cậu! Cậu khoẻ không? Lâu rồi không gặp…"

    The last one is the whole point of `prosody`: three sentences, three
    different pauses after them, and only one of them ends on a rise.
    """
    from pathlib import Path

    result = synthesize(language, text, speaking_rate, emotion, expressiveness)
    Path(output).write_bytes(result)
    print(f"wrote {output} ({len(result) / 1e6:.1f} MB)")
