"""Fine-tuning Seed-VC on one person, so the voice stops being a guess.

    clips ──► split_clips ──► /models/datasets/<voice>/*.wav ──► Trainer
                                                                    │
                              /models/voices/<voice>/<mode>/ft_model.pth
                                                             config.yml
                                                                    │
                                            VoiceConverter(voice="<voice>")

Everything else in this app is zero-shot: the model is shown twenty seconds of
somebody and asked to be them for the next four minutes. That is a remarkable
thing to be able to do and it has a ceiling, and the ceiling is exactly what
people describe when they say the result is *close* but still sounds like a
machine wearing a voice. Twenty seconds does not contain the person's whole
register, their breathiness at the bottom of it, what their consonants do when
they are quick — so the model fills those in from the average of everyone it
was trained on, and the average of everyone is what an AI voice sounds like.

Fine-tuning replaces the guess with data. Upstream's `train.py` is used as a
library, unmodified, and it is worth being precise about what it does, because
it is not what "training a voice model" usually means:

* it fine-tunes the **DiT** — the diffusion decoder — and its length
  regulator, and nothing else. The content encoder (Whisper), the speaker
  encoder (CAM++) and the vocoder stay exactly as they were;
* it needs **no transcripts and no labels**, only audio of one speaker. The
  training signal is self-supervised: each clip is put through OpenVoice's tone
  converter into a random other timbre, and the model is asked to reconstruct
  the original from the altered version's content plus the original's speaker
  embedding. So it learns to move *anything* into this speaker;
* a few minutes of audio and a few hundred steps is the working range. This is
  adaptation, not training from scratch — an hour of audio is not ten times
  better than five minutes, and 5000 steps is not five times better than 500.
  It is much easier to overfit a voice here than to underfit it.

**Two profiles per voice at most, and they are not interchangeable.** `speech`
and `singing` are different checkpoints at different sample rates with
different architectures, so a profile is stored per mode and `voices.py` keeps
that a fact about the filesystem.

**This is an operator tool, not an endpoint.** Nothing in `api.py` starts a
training run: it is ten minutes of a GPU, it needs audio nobody has agreed to
rate limit, and the consent question for "keep this person's voice on the
server permanently" is a different question from the one the submit form asks.
A run is started from a shell by whoever pays for the GPU:

    modal run -m modal_app.training --voice mai --audio ./mai.wav
    modal run -m modal_app.training --voice mai --audio ./clips --mode singing

and jobs then ask for it by name — `voice_profile=mai` on `/submit`, which is
free to say and does nothing if no such profile exists for that mode.
"""

from __future__ import annotations

from . import voices
from .app import MODEL_DIR, app, model_vol
from .conversion import SEED_VC_DIR, vc_image

# `tqdm` is the only thing upstream's training path needs that the inference
# path does not; everything else it imports — openvoice's tone converter, RMVPE,
# the optimizers — is already in the repo and already satisfied by
# `SEED_VC_REQUIREMENTS`.
training_image = vc_image.pip_install("tqdm")

# Where a run's working files go, both under `MODEL_DIR` so they survive a
# preempted container. `runs/` is `log_dir` in every seed-vc config; `datasets/`
# is ours.
DATASET_SUBDIR = "datasets"
RUNS_SUBDIR = "runs"

# Which architecture each mode fine-tunes, as the preset config upstream ships
# for it. These are the same architectures `inference.load_models` builds — the
# file names match the configs it pulls from HF — which is what makes a
# checkpoint trained here loadable there.
CONFIGS = {
    "speech": "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    "singing": "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
}

# The checkpoint each run *starts* from, named rather than left to the config.
#
# Both configs carry a `pretrained_model` of their own and the singing one
# names an older revision (`ft_ema`) than the one this app converts with
# (`ft_ema_v2`). Starting a fine-tune from a different checkpoint than the one
# it will be compared against is a needless variable, so these are copied from
# `inference.load_models` and passed explicitly.
PRETRAINED = {
    "speech": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "singing": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
}
PRETRAINED_REPO = "Plachta/Seed-VC"

# Clip length for the dataset. `data/ft_dataset.py` silently skips anything
# under 1s or over 30s, so the window sits well inside both: long enough to
# hold a phrase, short enough that a batch of two fits on the GPU at 44.1 kHz.
CLIP_MIN_SEC = 2.0
CLIP_MAX_SEC = 12.0
# Under this much usable audio there is not enough of the speaker to adapt to,
# and the run would spend ten GPU minutes memorising one sentence.
MIN_TOTAL_SEC = 30.0
# Past this the dataset stops improving and the epochs just get longer.
MAX_TOTAL_SEC = 20 * 60.0

# Steps. 500 is roughly ten minutes on an A10G and is where a personal voice
# has already moved most of the distance it is going to move; the ceiling is
# there because past it the model starts reproducing the recording rather than
# the speaker.
DEFAULT_STEPS = 500
MIN_STEPS = 100
MAX_STEPS = 5000
# Two clips at a time. Not a tuning choice so much as what fits: at 44.1 kHz
# with a 30s context the singing config's own default is 1.
BATCH_SIZE = 2

# Ten minutes of GPU for the default run, and room for a cold start that has to
# fetch four sets of pretrained weights before it can begin.
TRAINING_TIMEOUT = 3600


class TrainingError(ValueError):
    """Not enough audio, an unusable name, or a mode with no config."""


def clamp_steps(steps: int | None) -> int:
    try:
        value = DEFAULT_STEPS if not steps else int(steps)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, value))


def split_clips(audio, sample_rate: int) -> list:
    """One long recording as training clips, cut at the quietest frames.

    `audio_utils.split_at_silence` already knows how to find a quiet frame
    inside a window and is reused rather than reimplemented; the only
    difference is the scale — a training clip is seconds where a conversion
    chunk is half a minute — and that it does not overlap, because two clips
    sharing audio is the same audio counted twice in an epoch.

    Silence-only stretches are dropped: a dataset of room tone teaches the
    model that this speaker is a room.
    """
    import numpy as np

    from .audio_utils import split_at_silence

    pieces = split_at_silence(
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        target_sec=(CLIP_MIN_SEC + CLIP_MAX_SEC) / 2,
        max_sec=CLIP_MAX_SEC,
        min_sec=CLIP_MIN_SEC,
        search_sec=(CLIP_MAX_SEC - CLIP_MIN_SEC) / 4,
        overlap_sec=0.0,
    )
    kept = []
    for piece in pieces:
        if len(piece) < CLIP_MIN_SEC * sample_rate:
            continue
        peak = float(np.abs(piece).max())
        # The same relative floor `reference.py` judges a frame by. Absolute
        # thresholds are wrong here for the same reason: a quietly recorded
        # sample is not a silent one.
        if peak < 0.02:
            continue
        kept.append(piece)
    return kept


def build_dataset(clips: list, sample_rate: int, directory: str) -> float:
    """Write the training wavs and return how many seconds of audio there are.

    Every clip is cleaned on the way in — `reference.clean`, the same
    correction the conversion reference gets. It matters more here than there:
    a fine-tune learns whatever is consistent across the dataset, and a hum
    that is present in every clip is the most consistent thing in it.
    """
    import os

    from .audio_utils import encode_wav
    from .reference import clean

    os.makedirs(directory, exist_ok=True)
    total = 0.0
    written = 0
    for source in clips:
        for piece in split_clips(clean(source, sample_rate), sample_rate):
            if total >= MAX_TOTAL_SEC:
                break
            path = os.path.join(directory, f"clip_{written:05d}.wav")
            with open(path, "wb") as out:
                out.write(encode_wav(piece, sample_rate))
            total += len(piece) / sample_rate
            written += 1
    if total < MIN_TOTAL_SEC:
        raise TrainingError(
            f"need at least {MIN_TOTAL_SEC:.0f}s of usable audio to train a voice, "
            f"got {total:.1f}s across {written} clip(s)"
        )
    return total


@app.function(
    image=training_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_vol},
    timeout=TRAINING_TIMEOUT,
    # One run at a time. Training is the only thing in this app that would
    # rather have the whole GPU than share it with four conversions.
    max_containers=1,
    retries=0,
)
def train_voice(
    voice: str,
    clips: list,
    mode: str = "speech",
    steps: int = DEFAULT_STEPS,
) -> dict:
    """Fine-tune `mode`'s checkpoint on `clips` and save it as profile `voice`.

    `clips` is a list of encoded audio files — one long recording is fine, and
    so are twenty short ones. They are decoded at the config's own sample rate,
    cleaned, cut and written to the Volume, and then upstream's `Trainer` runs
    over that directory exactly as it would over anything else.

    Returns what was done, for the log and for whoever started it.
    """
    import os
    import shutil
    import sys
    import time

    import yaml

    from .audio_utils import decode_audio

    voices.check_name(voice)
    if mode not in CONFIGS:
        raise TrainingError(f"mode must be one of {tuple(CONFIGS)}, got {mode!r}")
    total_steps = clamp_steps(steps)

    # Same chdir as `VoiceConverter.load`, and for the same reason: seed-vc
    # writes its downloads and its runs into paths relative to the working
    # directory, and on the Volume is where those should land.
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.chdir(MODEL_DIR)
    if SEED_VC_DIR not in sys.path:
        sys.path.insert(0, SEED_VC_DIR)

    config_path = os.path.join(SEED_VC_DIR, CONFIGS[mode])
    sample_rate = int(yaml.safe_load(open(config_path))["preprocess_params"]["sr"])

    dataset = os.path.join(MODEL_DIR, DATASET_SUBDIR, voice, mode)
    shutil.rmtree(dataset, ignore_errors=True)
    seconds = build_dataset(
        [decode_audio(clip, sample_rate) for clip in clips], sample_rate, dataset
    )
    print(f"[train_voice] {voice}/{mode}: {seconds:.1f}s of audio, {total_steps} steps")

    from hf_utils import load_custom_model_from_hf  # seed-vc, on PYTHONPATH
    from train import Trainer

    pretrained = load_custom_model_from_hf(PRETRAINED_REPO, PRETRAINED[mode], None)
    run_name = f"{voice}_{mode}"
    started = time.time()
    Trainer(
        config_path=config_path,
        pretrained_ckpt_path=pretrained,
        data_dir=dataset,
        run_name=run_name,
        batch_size=BATCH_SIZE,
        steps=total_steps,
        # One save, at the end. `save_interval` writes intermediate checkpoints
        # that are ~2 GB each and that nothing here would ever load; setting it
        # past the step count is how upstream is told not to.
        save_interval=total_steps + 1,
        num_workers=0,
    ).train()
    elapsed = time.time() - started

    # `Trainer` saves to `<log_dir>/<run_name>/ft_model.pth` and copies its
    # config in beside it. Both are moved to the profile directory under the
    # names `voices.py` expects, so nothing downstream has to know that a run
    # directory ever existed.
    produced = os.path.join(MODEL_DIR, RUNS_SUBDIR, run_name)
    checkpoint, config = voices.files(MODEL_DIR, voice, mode)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    shutil.copyfile(os.path.join(produced, voices.CHECKPOINT_NAME), checkpoint)
    shutil.copyfile(os.path.join(produced, os.path.basename(config_path)), config)
    # The dataset and the run directory are working files, and a voice profile
    # is not a reason to keep somebody's audio on a server indefinitely — the
    # rest of this app expires user audio after six hours.
    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(produced, ignore_errors=True)
    model_vol.commit()

    result = {
        "voice": voice,
        "mode": mode,
        "steps": total_steps,
        "seconds": round(seconds, 1),
        "minutes": round(elapsed / 60, 1),
    }
    print(f"[train_voice] saved {checkpoint} ({result})")
    return result


@app.function(image=training_image, volumes={MODEL_DIR: model_vol}, timeout=120)
def list_voices() -> dict:
    """Every trained profile on the Volume, as `{voice: [mode, …]}`."""
    model_vol.reload()
    return voices.profiles(MODEL_DIR)


@app.local_entrypoint()
def train(
    voice: str,
    audio: str,
    mode: str = "speech",
    steps: int = DEFAULT_STEPS,
) -> None:
    """Train a voice profile from local audio.

        modal run -m modal_app.training --voice mai --audio ./mai.wav
        modal run -m modal_app.training --voice mai --audio ./clips --mode singing

    `--audio` is a file or a directory of them. A directory is read one level
    deep and in name order, which is the only thing that makes a run
    reproducible when somebody adds a clip and runs it again.
    """
    from pathlib import Path

    source = Path(audio)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and not p.name.startswith("."))
    else:
        files = [source]
    if not files:
        raise SystemExit(f"no audio in {audio}")

    print(f"training {voice}/{mode} on {len(files)} file(s)")
    result = train_voice.remote(
        voice=voice,
        clips=[path.read_bytes() for path in files],
        mode=mode,
        steps=steps,
    )
    print(result)
    print(f"use it with: modal run -m modal_app.conversion --voice {voice} --mode {mode} …")
