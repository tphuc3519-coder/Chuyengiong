"""Modal App, Volumes, Dict and shared container images.

Everything stateful lives here so the rest of the package can import a single
source of truth. No business logic in this module.
"""

import os

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
# Rate limit windows, keyed by a salted hash of the client address (Phase 4).
rate_dict = modal.Dict.from_name("vc-ratelimit", create_if_missing=True)

# Deployment configuration, all optional.
#
#   ALLOWED_ORIGINS  comma separated origins the browser may call from; empty
#                    means "*", which is where this starts before a Vercel
#                    domain exists (`api.allowed_origins`)
#   RATE_LIMIT_SALT  makes the rate limit's client hash unguessable as well as
#                    non-reversible (`ratelimit._salt`)
#   WATERMARK        set to 0/false/no/off to ship unwatermarked output; empty
#                    means on, which is the default (`watermark.enabled`)
#   JOBS_PER_HOUR    a positive integer puts the per-IP cap back; empty means
#                    no cap, which is where this now starts (`ratelimit.max_jobs`)
#   HF_TOKEN         a Hugging Face token whose account has accepted Stable
#                    Audio Open's licence. Only the beat generator needs it, and
#                    only generated beats need the generator — everything else
#                    on the deployment works without it (`beatgen.BeatGenerator`)
#   BEAT_GENERATOR   set to 1/true/yes/on to ship the beat generator. Off by
#                    default because its image does not build yet, and a broken
#                    image fails the whole deploy rather than its own function
#                    (`beatgen.enabled`)
CONFIG_KEYS = (
    "ALLOWED_ORIGINS",
    "RATE_LIMIT_SALT",
    "WATERMARK",
    "JOBS_PER_HOUR",
    "HF_TOKEN",
    "BEAT_GENERATOR",
)


def config_secret() -> modal.Secret:
    """Config from the machine running `modal deploy`, as container env vars.

    `Secret.from_dict` rather than `Secret.from_name`: a named secret has to
    exist before it can be looked up, so a fresh clone running its first deploy
    would fail on config that is entirely optional. This reads whatever the
    deploy environment happens to set — the workflow passes both through — and
    an unset key arrives as an empty string, which both readers treat as "not
    configured".
    """
    return modal.Secret.from_dict({key: os.environ.get(key, "") for key in CONFIG_KEYS})


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
    # copy=True: watermark.py and conversion.py both chain further build steps
    # (pip_install, run_commands) onto this image, which Modal forbids after a
    # mounted add_local_* — so bake the source in instead of mounting it.
    .add_local_python_source("modal_app", copy=True)
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
