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

One engine would have been better. Japanese needed a second, and then a third:
`ACCENT_MARKS` is the account of why. Reading Japanese without being able to
say where the pitch falls is not an accent problem, it is a wrong-word problem,
and `OpenJTalkSynthesizer` is the engine that does not have it.

**Neither of them is read flat.** Both speak in one fixed speaker with no
emotion conditioning, so what they are given instead is a plan: `prosody.py`
turns the text into a `Beat` per sentence — pace, height, level and the length
of the silence after it — and this module spends it on whichever knobs its
engine has. That plan is the difference between a page read out and a page got
through, and it is the only part of the reading either engine could not have
worked out from the sentence in front of it: it is handed one at a time and
cannot see the paragraph.

Note which rate goes to the engine. `Beat.synth_rate`, never `Beat.rate`: the
pitch is applied by resampling the result, which changes its length as well,
and `synth_rate` is the pace that comes out right once it has.

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
OPENJTALK = "openjtalk"


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
        # Open JTalk rather than Kokoro, and the reason is `ACCENT_MARKS`: 箸
        # and 橋 are both `hashi` and only the pitch separates them, Kokoro-82M
        # v1.0 has no way of being told which, and Open JTalk reads the accent
        # out of a dictionary. `KokoroSynthesizer` is still here and still
        # works — `modal run -m modal_app.tts --engine kokoro` is the A/B —
        # because this is a trade rather than a free win: the HTS voice is far
        # less natural, and what makes that bearable is the thing this module
        # opens by saying, that the synthetic voice is a stand-in and Seed-VC
        # replaces the timbre one step later. Correct beats pretty here.
        engine=OPENJTALK,
        # ~700 Japanese characters is the same two to three minutes of speech
        # the Latin limit buys, and 80 per segment keeps the phoneme string
        # well under the 510 Kokoro truncates at.
        max_chars=700,
        segment_max_chars=80,
        romaji_input=True,
        # Kept for the A/B: which engine reads is `engine`'s to say, and these
        # are what `KokoroSynthesizer` needs when it is the one asked.
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

# Open JTalk, which reads Japanese pitch accent out of a dictionary. See
# `ACCENT_MARKS` for why that matters and why Kokoro cannot be told it.
OPENJTALK_SPEC = "pyopenjtalk==0.4.1"

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
                #
                # `synth_rate`, not `rate`: the transposition `prosody.shape`
                # applies afterwards is a resample, so it changes the length
                # too, and this is the pace that comes out right once it has.
                # Its own clamp is the outer limit — at the far end of the
                # slider it can bite, and then the sentence keeps its pitch and
                # loses a little of its timing, which is the right way round.
                self.model.speaking_rate = clamp_speaking_rate(beat.synth_rate)
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


# The image both Japanese engines share, because they share a dependency:
# `misaki[ja]` requires `pyopenjtalk`, so installing Kokoro's front end already
# installs Open JTalk. One image, one build, and the A/B costs nothing.
#
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
japanese_image = (
    base_image.pip_install(
        "kokoro==0.9.4",
        "misaki[ja]==0.9.4",
        "unidic-lite==1.0.8",
        # Pinned rather than left as `misaki[ja]`'s unversioned dependency,
        # because it is now an engine here rather than something a front end
        # happens to pull in. PyPI ships it as an sdist only — no wheels for
        # any version — so this compiles open_jtalk and hts_engine during the
        # build. It already did, silently, as misaki's dependency.
        OPENJTALK_SPEC,
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
    # Read a word at build time. Open JTalk's dictionary ships inside the
    # package, but `_lazy_init` fetches it from a GitHub release if it is ever
    # missing — which would be a download on a cold container, on a request,
    # over a network the container may not have. Touching it here settles that
    # in a layer, and a broken chain fails the deploy instead of the first
    # Japanese job.
    .run_commands(
        "python -c \"import pyopenjtalk; print(pyopenjtalk.g2p('今日はいい天気ですね。'))\""
    )
    .env({"HF_HOME": MODEL_DIR})
)

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000

# --- Japanese pitch accent ------------------------------------------------
#
# 箸 and 橋 are both `hashi`. 雨 and 飴 are both `ame`. What separates them is
# where the pitch falls, and getting it wrong is not an accent — it is a
# different word. So how the accent reaches the model is the single most
# important thing about reading Japanese here.
#
# `misaki` has two Japanese front ends and `kokoro.KPipeline` builds the first
# one, because `JAG2P()` defaults to `version='cutlet'`:
#
#   1st gen  cutlet -> fugashi -> mecab -> unidic-lite.  No accent marks at
#            all: the phoneme string says which sounds, never which pitch, and
#            the model is left to guess.
#   2nd gen  `version='pyopenjtalk'`. `pyopenjtalk.run_frontend` returns an
#            accent nucleus per word out of Open JTalk's dictionary, and the
#            G2P appends a pitch track to the phoneme string — one character
#            per phoneme, `_` low, `-` mid, `^` the fall.
#
# The second is plainly the one to want. Whether it can be used at all is a
# question about the *checkpoint*, not about the library: `KModel.forward`
# maps phonemes through `self.vocab` and **silently drops** everything it has
# no id for, so handing the pitch track to a checkpoint that was not trained
# with it does not fail — it quietly deletes the marks and pronounces whatever
# is left, and `j`, the track's own filler character, is the IPA phoneme /j/.
# That is a worse reading than not trying.
#
# So it is asked rather than assumed, the same way `Synthesizer.load` asks
# whether MMS wants romanised input instead of hoping. The answer is in
# `model.vocab`.
ACCENT_MARKS = ("_", "-", "^")
# Read to the log on load so the G2P chain is visible in a container's output
# without one word of anybody's text passing through it.
ACCENT_PROBE = "今日はいい天気ですね。"
# Kokoro truncates a phoneme string past this, with a warning and no error.
# `segment_max_chars` is set well under it; this is here to explain the margin.
KOKORO_MAX_PHONEMES = 510


@app.cls(
    image=japanese_image,
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
        # What this class actually needs, rather than what the language happens
        # to be configured to use: Japanese reads through Open JTalk now, and
        # `--engine kokoro` has to still be able to get here to be compared to.
        if not spec.kokoro_code:
            raise TtsError(f"{self.language} has no Kokoro voice to read it")
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.spec = spec
        # Weights, config and the voice pack all arrive through
        # `huggingface_hub`, so HF_HOME on the Volume is what keeps a second
        # container from downloading them again.
        self.pipeline = KPipeline(lang_code=spec.kokoro_code, repo_id=KOKORO_REPO)
        self.accent = self._use_accent_g2p()
        model_vol.commit()
        print(
            f"[KokoroSynthesizer] language={self.language} voice={spec.voice} accent={self.accent}"
        )
        print(f"[KokoroSynthesizer] {ACCENT_PROBE} -> {self._phonemes(ACCENT_PROBE)}")

    def _use_accent_g2p(self) -> bool:
        """Swap in misaki's accent-carrying front end, if the checkpoint reads it.

        Returns whether it was swapped, and never raises: a reading without
        accent marks is the reading this shipped with, and it is much better
        than a container that will not start.

        The gate is `model.vocab`, because that is what decides whether the
        pitch track becomes input ids or becomes nothing. See `ACCENT_MARKS`.
        """
        model = getattr(self.pipeline, "model", None)
        vocab = getattr(model, "vocab", None)
        if not vocab:
            print("[KokoroSynthesizer] no vocab to check; leaving the G2P alone")
            return False

        missing = [mark for mark in ACCENT_MARKS if mark not in vocab]
        if missing:
            # Expected for Kokoro-82M v1.0, which `KOKORO_REPO` points at: it
            # was trained on the first generation's phonemes. Said out loud
            # rather than passed over, because it is the ceiling on how right
            # the Japanese can be here, and the next person to ask why should
            # find the answer in the log.
            print(
                f"[KokoroSynthesizer] {KOKORO_REPO} has no id for {missing} — it cannot be "
                "told where the pitch falls, so the accent is the model's guess"
            )
            return False

        try:
            from misaki import ja
        except ImportError:
            print("[KokoroSynthesizer] misaki[ja] is not importable; leaving the G2P alone")
            return False
        self.pipeline.g2p = ja.JAG2P(version="pyopenjtalk")
        return True

    def _phonemes(self, text: str) -> str:
        """What the G2P makes of `text`, for the log. Never user text."""
        try:
            result = self.pipeline.g2p(text)
        except Exception as exc:  # a front end that cannot read the probe
            return f"<{type(exc).__name__}: {exc}>"
        return result[0] if isinstance(result, tuple) else str(result)

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
        and nothing else, so the pace is set per sentence here and the pitch and
        the level are `prosody.shape`'s to apply. That is the same division as
        MMS — `noise_scale` is the only thing MMS adds — and the reading comes
        out the same shape either way.

        This is the branch that had to be got right twice. Japanese spends its
        meaning on mora length, and the first version of `prosody.shape`
        transposed every sentence with a phase vocoder whose window was longer
        than the distinctions involved; it read the words wrong. `_transpose`
        carries the account of it.
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
                    # `synth_rate` for the same reason as MMS: the resample in
                    # `prosody.shape` shortens what it raises, and this is what
                    # pays for it.
                    speed=clamp_speaking_rate(beat.synth_rate),
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


# Open JTalk's HTS engine returns 48 kHz, and returns it in int16 units rather
# than in [-1, 1].
OPENJTALK_SAMPLE_RATE = 48000
OPENJTALK_SCALE = 32768.0


@app.cls(
    image=japanese_image,
    # No GPU and no Volume: the HTS voice and the dictionary are both inside
    # the package, so there is nothing to fetch and nothing to cache. It is by
    # far the cheapest of the three engines to start.
    cpu=2,
    memory=2048,
    scaledown_window=300,
    timeout=900,
    max_containers=4,
)
class OpenJTalkSynthesizer:
    """Japanese, read with the accent a dictionary says rather than a guess.

    The engine is old — HTS, the statistical parametric kind, and it sounds it.
    It is here because of what this module says in its first paragraph: the
    synthetic voice is a stand-in and Seed-VC replaces the timbre a step later,
    so a checkpoint is chosen for correctness rather than for beauty. And the
    thing Open JTalk is correct about is the one thing Japanese cannot afford
    to have wrong. 箸 and 橋 are both `hashi`; 雨 and 飴 are both `ame`. Where
    the pitch falls is which word it is, `pyopenjtalk.run_frontend` reads that
    out of Open JTalk's dictionary, and its synthesis backend renders it.

    What it costs is naturalness, and the buzz of an HTS excitation is exactly
    the part of the signal Seed-VC throws away. What it buys is words.

    Two conveniences fall out of it, both better than the workarounds the other
    engines need: `speed` and `half_tone` are synthesis parameters, so the pace
    and the pitch of a sentence are set before a sample exists — no resample,
    no formants dragged along, no length to pay back. `prosody.shape` is left
    with the level alone, which is why it is called with `engine_pitch=True`
    and why this is the one engine handed `Beat.rate` rather than
    `Beat.synth_rate`.
    """

    language: str = modal.parameter(default=JAPANESE)

    @modal.enter()
    def load(self) -> None:
        import pyopenjtalk

        if self.language != JAPANESE:
            raise TtsError(f"Open JTalk reads Japanese, not {self.language}")
        self.spec = spec_for(self.language)
        # Also what pulls the dictionary into memory, so the first request does
        # not pay for it. The probe is ours, never the user's text.
        print(f"[OpenJTalkSynthesizer] {ACCENT_PROBE} -> {pyopenjtalk.g2p(ACCENT_PROBE)}")

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaking_rate: float = DEFAULT_SPEAKING_RATE,
        emotion: str = prosody.DEFAULT_EMOTION,
        expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
    ) -> bytes:
        """Text in, 16-bit PCM wav out at 48 kHz. Same contract as the others."""
        import time

        import numpy as np
        import pyopenjtalk

        spoken_text = check_text(text, self.language)
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
            # `rate`, not `synth_rate`: nothing is resampled afterwards, so
            # there is nothing to pay back. `half_tone` is semitones, which is
            # the unit `Beat.pitch` is already in.
            audio, sample_rate = pyopenjtalk.tts(
                beat.text,
                speed=clamp_speaking_rate(beat.rate),
                half_tone=float(beat.pitch),
            )
            audio = np.asarray(audio, dtype=np.float32) / OPENJTALK_SCALE
            if not audio.size:
                print(f"[OpenJTalkSynthesizer] segment {i}/{len(beats)} produced no audio")
                continue
            spoken.append(prosody.shape(audio, sample_rate, beat, engine_pitch=True))
            pauses.append(beat.pause_sec)

        wav = _join(spoken, OPENJTALK_SAMPLE_RATE, pauses)
        print(
            f"[OpenJTalkSynthesizer] {len(beats)} segment(s) in {time.time() - started:.1f}s "
            f"(rate={rate:.2f} emotion={prosody.clean_emotion(emotion)} "
            f"expressiveness={prosody.clamp_expressiveness(expressiveness):.2f})"
        )
        return wav


# Which class reads which engine. A dict rather than a chain of conditionals
# for the same reason `pipeline.PIPELINES` is one: a missing entry is a
# KeyError here instead of a language quietly read by the wrong model.
ENGINES = {
    MMS: Synthesizer,
    KOKORO: KokoroSynthesizer,
    OPENJTALK: OpenJTalkSynthesizer,
}


def synthesize(
    language: str,
    text: str,
    speaking_rate: float,
    emotion: str = prosody.DEFAULT_EMOTION,
    expressiveness: float = prosody.DEFAULT_EXPRESSIVENESS,
    engine: str = "",
) -> bytes:
    """Read `text` with whichever engine speaks `language`, on its own container.

    The one place the engine split is resolved. `pipeline.py` asks for a
    language and gets audio back; which model produced it is this module's
    business, and adding a third engine should not reach the pipeline at all.
    The same is now true of how it is read: every engine takes the same style,
    because the style is `prosody`'s and not any model's.

    `engine` overrides the language's own choice. It exists for the smoke test
    and for nothing else — no request can set it — because Japanese has two
    engines that are a real trade against each other and the only way to judge
    that trade is to hear both.
    """
    spec = spec_for(language)
    return ENGINES[engine or spec.engine](language=language).synthesize.remote(
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
    engine: str = "",
) -> None:
    """Standalone smoke test: text in, one wav out. No reference, no GPU.

    modal run -m modal_app.tts --text "Xin chào."
    modal run -m modal_app.tts --language jpn --text "今日はいい天気ですね。"
    modal run -m modal_app.tts --emotion cheerful \
        --text "Chào cậu! Cậu khoẻ không? Lâu rồi không gặp…"

    The third one is the whole point of `prosody`: three sentences and three
    different pauses after them.

    And the comparison the Japanese engine choice rests on, which is a trade
    and wants ears rather than an argument — dictionary accent against a
    guess, an old voice against a natural one:

        --language jpn --output openjtalk.wav --text "箸と橋、雨と飴。"
        --language jpn --output kokoro.wav --engine kokoro --text "箸と橋、雨と飴。"

    Both are the *input* to the conversion rather than the product, so what
    settles it is which one comes out of Seed-VC saying the right words.
    """
    from pathlib import Path

    result = synthesize(language, text, speaking_rate, emotion, expressiveness, engine)
    Path(output).write_bytes(result)
    print(f"wrote {output} ({len(result) / 1e6:.1f} MB)")
