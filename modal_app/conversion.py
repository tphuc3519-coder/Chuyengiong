"""Seed-VC zero-shot voice conversion on a Modal GPU container.

Ported from `Plachtaa/seed-vc` at the commit pinned below. Model loading is
reused verbatim (`inference.load_models`); what we add on top is the split that
the plan requires and upstream does not have:

* the reference is encoded **once** and reused for every chunk, so timbre
  cannot drift mid-song;
* `semitone_shift` is a required argument computed for the whole track by the
  caller (Phase 5), never auto-detected per chunk — per-chunk detection is the
  bug that makes the voice jump between verses;
* the source is cut at silence into overlapping chunks and rejoined with an
  equal-power crossfade, which is what keeps an 8 minute song from running the
  A10G out of memory.

Three things were added to it after the fact, all for the same complaint —
that the output sounded converted rather than sung — and all named here because
none of them is visible from the class below:

* the reference is **cleaned** before it is encoded (`reference.py`). Seed-VC
  reproduces a sample, not a voice: room tone in the reference is room tone
  fused into every syllable of the result, and that is most of what people
  mean when they say a result sounds artificial;
* the speaker embedding is **averaged over every window of the reference worth
  using**, not taken from one. A window is a sample of a speaker; the average
  of three is a better estimate of the speaker than the best single one;
* a voice can be **fine-tuned** (`training.py`) and loaded here by name. Zero
  shot is one twenty-second look at somebody; a profile is a few hundred steps
  of actually learning them, and it is the only thing in this file that changes
  the model rather than what the model is shown.

Two things differ from the plan doc, both deliberate:

* `mode` is a class parameter, not an argument of `convert()`. The two modes
  need different checkpoints, sample rates and vocoders, so `VoiceConverter(
  mode="singing")` lets `@modal.enter()` load exactly one model set and keeps
  each mode's containers warm separately.
* reference audio is capped at 20s, not 30s — see `audio_utils`.

Smoke test (needs Modal credentials and a GPU):

    modal run -m modal_app.conversion --source song.mp3 --reference voice.wav
"""

# NB: no `from __future__ import annotations` here — modal.parameter() reads
# the raw class annotation and cannot resolve a stringified one.
from dataclasses import dataclass
from typing import Any

import modal

from . import reference as reference_audio
from . import voices
from .app import MODEL_DIR, app, base_image, model_vol
from .audio_utils import (
    CHUNK_OVERLAP_SEC,
    DEFAULT_CFG_RATE,
    check_mode,
    check_source,
    clamp_cfg_rate,
    clamp_diffusion_steps,
    clamp_semitone_shift,
    crossfade_concat,
    decode_audio,
    encode_wav,
    split_at_silence,
)

SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc"
# Pinned, not `main`: seed-vc changes its inference API between versions.
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
SEED_VC_DIR = "/opt/seed-vc"

# Upstream's requirements.txt at that commit, minus:
#  * its three `--pre --index-url .../nightly/cu126` torch lines — the same file
#    also pins the 2.4.0 release builds, which is what base_image installs;
#  * gradio / FreeSimpleGUI / sounddevice — the GUIs we do not run;
#  * jiwer / modelscope / funasr / resemblyzer — eval and training only.
# torch is repeated here so the resolver cannot drag in a different build while
# satisfying descript-audio-codec.
SEED_VC_REQUIREMENTS = [
    "torch==2.4.0",
    "torchvision==0.19.0",
    "torchaudio==2.4.0",
    "accelerate",
    "scipy==1.13.1",
    "librosa==0.10.2",
    "huggingface-hub>=0.28.1",
    "munch==4.0.0",
    "einops==0.8.0",
    "descript-audio-codec==1.0.0",
    "pydub==0.25.1",
    "transformers==4.46.3",
    "soundfile==0.12.1",
    "numpy==1.26.4",
    "hydra-core==1.3.2",
    "pyyaml",
    "python-dotenv",
]

# `descript-audio-codec`, which seed-vc pins, pulls in `descript-audiotools`,
# which caps protobuf at `<3.20` for a tensorboard logger that only training
# touches. Modal's own agent runs inside this image and reads
# `api_pb2.<Enum>.ValueType`, an attribute protobuf grew in 3.20 — under that cap
# every container dies with
#
#     AttributeError: Enum VolumeFsVersion has no value defined for name 'ValueType'
#
# before a line of this module runs, which reads as the GPU never starting rather
# than as a dependency conflict. The separation image escapes it only by not
# depending on DAC.
#
# A separate build step, deliberately: inside `SEED_VC_REQUIREMENTS` the resolver
# would have to satisfy the cap, and here it lands after it instead. pip prints a
# conflict warning, which is the intended outcome — the upper bound is Modal's own.
PROTOBUF_SPEC = "protobuf>=3.20,<7"

vc_image = (
    base_image.run_commands(
        f"git clone {SEED_VC_REPO} {SEED_VC_DIR}",
        f"cd {SEED_VC_DIR} && git checkout {SEED_VC_COMMIT}",
    )
    .pip_install(*SEED_VC_REQUIREMENTS)
    .pip_install(PROTOBUF_SPEC)
    .env({"PYTHONPATH": SEED_VC_DIR, "HF_HOME": MODEL_DIR})
)

# mode -> whether the checkpoint is F0 conditioned. `singing` loads
# seed-uvit-whisper-base (44.1kHz, F0), `speech` loads
# seed-uvit-whisper-small-wavenet (22.05kHz). Both use a Whisper-small content
# encoder, which is what gives us ~99 languages; do not swap either for the
# xlsr tiny checkpoint just because it is faster.
F0_CONDITION = {"speech": False, "singing": True}

# Upstream's streaming constants (inference.py): seed-vc fits source+reference
# in one 30s context window and overlaps its internal windows by 16 mel frames.
CONTEXT_WINDOW_SEC = 30
INNER_OVERLAP_FRAMES = 16


@dataclass
class _Reference:
    """Everything derived from the voice sample, computed once per request."""

    mel: Any  # mel2 in upstream naming
    style: Any  # campplus speaker embedding
    prompt_condition: Any  # length-regulated content of the reference


@app.cls(
    image=vc_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,  # keep the container warm for 5 minutes
    timeout=1800,
    max_containers=4,  # a burst of jobs should queue, not multiply the bill
)
class VoiceConverter:
    mode: str = modal.parameter(default="singing")
    # Which fine-tuned voice this container is for, or "" for zero shot.
    #
    # A class parameter and not an argument of `convert`, for the same reason
    # `mode` is one: it decides which weights `@modal.enter()` loads, so a
    # container can only ever be one of them. Modal keys its containers on the
    # parameters, so `VoiceConverter(mode="speech", voice="mai")` and the
    # zero-shot one are two warm pools rather than one that keeps reloading.
    voice: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        """Runs once per container. Weights land in the Volume, not the image."""
        import os
        import sys
        from types import SimpleNamespace

        check_mode(self.mode)

        # seed-vc's hf_utils caches into "./checkpoints"; chdir so that lands on
        # the Volume and survives container restarts.
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.chdir(MODEL_DIR)
        if SEED_VC_DIR not in sys.path:
            sys.path.insert(0, SEED_VC_DIR)

        import inference as seed_vc  # upstream CLI module, used as a library

        self._seed_vc = seed_vc
        self.fp16 = True
        f0_condition = F0_CONDITION[self.mode]
        # A trained voice replaces the DiT checkpoint and nothing else: the
        # content encoder, the speaker encoder and the vocoder are all still
        # the pretrained ones, because that is all `train.py` fine-tunes.
        # `resolve` raises for a profile that was asked for and is not there —
        # running zero-shot instead would produce a confident wrong answer.
        profile = voices.resolve(MODEL_DIR, self.voice, self.mode)
        checkpoint, config = profile if profile else (None, None)
        (
            self.model,
            self.semantic_fn,
            self.f0_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            mel_fn_args,
        ) = seed_vc.load_models(
            SimpleNamespace(
                fp16=self.fp16,
                f0_condition=f0_condition,
                checkpoint=checkpoint,
                config=config,
            )
        )

        self.f0_condition = f0_condition
        self.device = seed_vc.device
        self.sr = mel_fn_args["sampling_rate"]
        self.hop_length = mel_fn_args["hop_size"]
        self.max_context_frames = self.sr // self.hop_length * CONTEXT_WINDOW_SEC
        self.overlap_samples = INNER_OVERLAP_FRAMES * self.hop_length

        model_vol.commit()
        print(
            f"[VoiceConverter] mode={self.mode} sr={self.sr} device={self.device} "
            f"voice={self.voice or 'zero-shot'}"
        )

    # --- pieces of upstream inference.main, split so state is reused ------

    def _semantic_features(self, wave_16k):
        """Whisper content features, chunked the way upstream chunks them.

        Whisper takes 30s at a time; anything longer is walked with a 5s
        overlap and the duplicated frames dropped.
        """
        import torch

        if wave_16k.size(-1) <= 16000 * 30:
            return self.semantic_fn(wave_16k)

        overlapping_time = 5
        features = []
        buffer = None
        traversed = 0
        while traversed < wave_16k.size(-1):
            if buffer is None:
                chunk = wave_16k[:, traversed : traversed + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, wave_16k[:, traversed : traversed + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            part = self.semantic_fn(chunk)
            features.append(part if traversed == 0 else part[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlapping_time
        return torch.cat(features, dim=1)

    def _style(self, wave_16k):
        """The campplus speaker embedding of one window of reference audio."""
        import torchaudio

        feat = torchaudio.compliance.kaldi.fbank(
            wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus_model(feat.unsqueeze(0))

    def _encode_reference(self, reference, extras=()) -> _Reference:
        """Encode the voice sample once; every chunk reuses the result.

        `reference` is the window the model is conditioned on — its mel, its
        content and its length all come from that one stretch of audio, and
        nothing else can supply them. `extras` are further windows of the same
        recording, and they contribute to one thing only: the speaker
        embedding, which is averaged over all of them.

        That split is the whole design. Averaging a mel or a content sequence
        across windows would be averaging two different sentences, which is
        meaningless. Averaging speaker embeddings is what speaker verification
        does at enrolment — the embedding of one window carries what is
        particular to that window as well as what is particular to the person,
        and only the second survives the mean.
        """
        import torch
        import torchaudio

        with torch.no_grad():
            ref = torch.tensor(reference).unsqueeze(0).float().to(self.device)
            ref_16k = torchaudio.functional.resample(ref, self.sr, 16000)

            content = self._semantic_features(ref_16k)
            mel = self.to_mel(ref.float())
            lengths = torch.LongTensor([mel.size(2)]).to(mel.device)

            style = self._style(ref_16k)
            for extra in extras:
                window = torch.tensor(extra).unsqueeze(0).float().to(self.device)
                style = style + self._style(torchaudio.functional.resample(window, self.sr, 16000))
            if extras:
                style = style / float(1 + len(extras))

            f0 = None
            if self.f0_condition:
                f0 = torch.from_numpy(self.f0_fn(ref_16k[0], thred=0.03)).float().to(self.device)
                f0 = f0[None]

            prompt_condition, *_ = self.model.length_regulator(
                content, ylens=lengths, n_quantizers=3, f0=f0
            )
        return _Reference(mel=mel, style=style, prompt_condition=prompt_condition)

    def _convert_chunk(
        self, chunk, reference: _Reference, semitone_shift, diffusion_steps, inference_cfg_rate
    ):
        """Convert exactly one chunk. Never call this from outside the class."""
        import numpy as np
        import torch
        import torchaudio

        with torch.no_grad():
            source = torch.tensor(chunk).unsqueeze(0).float().to(self.device)
            source_16k = torchaudio.functional.resample(source, self.sr, 16000)

            content = self._semantic_features(source_16k)
            mel = self.to_mel(source.float())
            # length_adjust is fixed at 1.0: we are not stretching time.
            target_lengths = torch.LongTensor([mel.size(2)]).to(mel.device)

            shifted_f0 = None
            if self.f0_condition:
                f0 = torch.from_numpy(self.f0_fn(source_16k[0], thred=0.03))
                f0 = f0.float().to(self.device)[None]
                shifted_f0 = f0.clone()
                if semitone_shift:
                    voiced = f0 > 1
                    shifted_f0[voiced] = self._seed_vc.adjust_f0_semitones(
                        f0[voiced], semitone_shift
                    )

            cond, *_ = self.model.length_regulator(
                content, ylens=target_lengths, n_quantizers=3, f0=shifted_f0
            )

            # Source and reference share one context window, so a long
            # reference shrinks the per-forward source window.
            max_source_window = self.max_context_frames - reference.mel.size(2)
            if max_source_window <= INNER_OVERLAP_FRAMES:
                raise RuntimeError(
                    "reference is too long for the model context window "
                    f"({reference.mel.size(2)} of {self.max_context_frames} frames)"
                )

            processed_frames = 0
            previous = None
            generated = []
            while processed_frames < cond.size(1):
                window = cond[:, processed_frames : processed_frames + max_source_window]
                is_last = processed_frames + max_source_window >= cond.size(1)
                condition = torch.cat([reference.prompt_condition, window], dim=1)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.fp16 else torch.float32,
                ):
                    target = self.model.cfm.inference(
                        condition,
                        torch.LongTensor([condition.size(1)]).to(reference.mel.device),
                        reference.mel,
                        reference.style,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )
                    target = target[:, :, reference.mel.size(-1) :]

                wave = self.vocoder_fn(target.float()).squeeze()[None, :]
                advance = target.size(2) - INNER_OVERLAP_FRAMES

                if is_last:
                    tail = wave[0].cpu().numpy()
                    generated.append(
                        tail
                        if previous is None
                        else self._seed_vc.crossfade(
                            previous.cpu().numpy(), tail, self.overlap_samples
                        )
                    )
                    break

                head = wave[0, : -self.overlap_samples].cpu().numpy()
                generated.append(
                    head
                    if previous is None
                    else self._seed_vc.crossfade(previous.cpu().numpy(), head, self.overlap_samples)
                )
                previous = wave[0, -self.overlap_samples :]
                processed_frames += advance

        return np.concatenate(generated)

    # --- the only entry point --------------------------------------------

    @modal.method()
    def convert(
        self,
        source_wav: bytes,
        reference_wav: bytes,
        semitone_shift: int,
        diffusion_steps: int = 0,
        inference_cfg_rate: float = DEFAULT_CFG_RATE,
    ) -> bytes:
        """Convert `source_wav` to the voice in `reference_wav`.

        `semitone_shift` is required and applies to the whole track: compute it
        once from the full vocal stem (Phase 5) and pass the same value here.
        `diffusion_steps` of 0 means "the default for this mode".

        `inference_cfg_rate` is classifier-free guidance: how far the model is
        pushed towards the reference and away from what it would have produced
        unconditioned. It reaches the product as "how much of the sample voice
        to take" — up, and the result is more clearly the target and more
        obviously processed; down, and it keeps more of whoever is on the
        source recording. Upstream's 0.7 is the middle of that and the default
        here.

        Returns 16-bit PCM wav bytes at the model's sample rate.
        """
        import time

        steps = clamp_diffusion_steps(diffusion_steps, self.mode)
        shift = clamp_semitone_shift(semitone_shift, self.mode)
        cfg_rate = clamp_cfg_rate(inference_cfg_rate)

        source = check_source(decode_audio(source_wav, self.sr), self.sr)
        # Cleaned, then cut into as many usable windows as the recording holds
        # — see `reference.py`. Everything after the first is only averaged
        # into the speaker embedding.
        reference, extras = reference_audio.prepare(decode_audio(reference_wav, self.sr), self.sr)

        chunks = split_at_silence(source, self.sr)
        print(
            f"[VoiceConverter] {len(source) / self.sr:.1f}s source -> {len(chunks)} chunks, "
            f"steps={steps} shift={shift:+d} cfg={cfg_rate:.2f} "
            f"reference={1 + len(extras)} window(s)"
        )

        started = time.time()
        encoded_reference = self._encode_reference(reference, extras)
        converted = []
        for i, chunk in enumerate(chunks, start=1):
            mark = time.time()
            converted.append(self._convert_chunk(chunk, encoded_reference, shift, steps, cfg_rate))
            print(
                f"[VoiceConverter] chunk {i}/{len(chunks)} "
                f"({len(chunk) / self.sr:.1f}s) in {time.time() - mark:.1f}s"
            )

        output = crossfade_concat(converted, self.sr, CHUNK_OVERLAP_SEC)
        print(f"[VoiceConverter] done in {time.time() - started:.1f}s")
        return encode_wav(output, self.sr)


@app.local_entrypoint()
def main(
    source: str,
    reference: str,
    output: str = "converted.wav",
    mode: str = "singing",
    semitone_shift: int = 0,
    diffusion_steps: int = 0,
    cfg_rate: float = DEFAULT_CFG_RATE,
    voice: str = "",
) -> None:
    """Standalone smoke test: two local audio files in, one wav out.

    `--voice` loads a profile `training.py` produced, which is the A/B that
    settles whether fine-tuning was worth the GPU minutes: same source, same
    reference, once with and once without.
    """
    from pathlib import Path

    result = VoiceConverter(mode=mode, voice=voice).convert.remote(
        source_wav=Path(source).read_bytes(),
        reference_wav=Path(reference).read_bytes(),
        semitone_shift=semitone_shift,
        diffusion_steps=diffusion_steps,
        inference_cfg_rate=cfg_rate,
    )
    Path(output).write_bytes(result)
    print(f"wrote {output} ({len(result) / 1e6:.1f} MB)")
