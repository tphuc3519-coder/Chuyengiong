"""Modal App, Volumes, Dict and shared container images.

Everything stateful lives here so the rest of the package can import a single
source of truth. No business logic in this module.
"""

import modal

APP_NAME = "voice-convert"

# Mount points inside containers.
MODEL_DIR = "/models"
DATA_DIR = "/data"

app = modal.App(APP_NAME)

# Model weights: written once, reused by every warm container.
model_vol = modal.Volume.from_name("vc-models", create_if_missing=True)
# User files: ephemeral, cleaned up by a scheduled job (Phase 2).
data_vol = modal.Volume.from_name("vc-data", create_if_missing=True)
# Job state machine records, keyed by job id (Phase 2).
job_dict = modal.Dict.from_name("vc-jobs", create_if_missing=True)

# Base image for GPU/audio work. Torch is pinned; `numpy<2` because the audio
# stack (librosa/soundfile) still trips over the numpy 2 ABI.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "huggingface_hub",
        "soundfile",
        "librosa",
        "numpy<2",
        "scipy",
        "fastapi[standard]",
    )
    .add_local_python_source("modal_app")
)

# The CPU image: web endpoints, the cleanup cron and the pipeline orchestrator.
# No torch and no GPU — the heavy work happens in the separation and conversion
# containers, which this one only calls. It carries ffmpeg (mixing and the final
# encode) and numpy (`audio_utils` is imported along the way), and nothing else,
# so a /status poll still starts in a couple of seconds.
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "numpy<2")
    .add_local_python_source("modal_app")
)
