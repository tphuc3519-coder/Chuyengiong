"""Orchestration: one job from uploaded bytes to a finished mp3.

    song:    input ──► separate ──► vocal ──► convert ──► mix ──► output.mp3
                            └────► instrumental ───────────┘
    speech:  input ─────────────────────────► convert ──► encode ──► output.mp3

These functions run on the small CPU image and do no audio work themselves:
separation and conversion happen in their own GPU containers, called with
`.remote()` so this one blocks on them without holding a GPU. `/submit` starts
a job with `.spawn()` and returns immediately.

The job's parameters are normalised **once**, in `clean_params`, before the job
record is written. Every clamp in the plan (semitone range per mode, diffusion
step range, vocal gain) is applied there rather than at the point of use, so
what the status endpoint reports and what the GPU receives cannot disagree.
"""

from __future__ import annotations

from . import jobs, storage
from .app import DATA_DIR, api_image, app, data_vol
from .audio_utils import clamp_diffusion_steps, clamp_semitone_shift
from .mixing import clamp_gain_db
from .separation import DEFAULT_SEPARATION_MODEL, check_model, safe_ext

# Artifact names on the Volume, matching the layout in `storage`. `input.mp3`
# is a label, not a claim about the container: the upload keeps whatever format
# it arrived in and `params["source_ext"]` carries the real extension through to
# the separator, which picks its decoder by file name.
INPUT = "input.mp3"
REFERENCE = "reference.wav"
VOCAL = "vocal.wav"
INSTRUMENTAL = "instrumental.wav"
CONVERTED = "converted.wav"
OUTPUT = "output.mp3"

# A 15 minute song at 44.1kHz is ~10 minutes of GPU work in the worst case;
# 30 minutes of wall clock leaves room for a cold start on both models.
PIPELINE_TIMEOUT = 1800

# Job records are read by a browser that is polling; keep failure text short
# enough to render and free of stack traces.
MAX_ERROR_CHARS = 400


def clean_params(mode: str, raw: dict | None = None) -> dict:
    """Validate and clamp everything `/submit` accepts. Pure, so tests cover it.

    Raises `jobs.JobError` for an unusable mode and `SeparationError` for an
    unknown separation model; every other field is clamped rather than
    rejected, because a slider that arrives out of range is a client bug, not
    something worth failing a paid GPU job over.
    """
    raw = dict(raw or {})
    jobs.check_mode(mode)
    conversion_mode = jobs.CONVERSION_MODE[mode]

    params = {
        "semitone_shift": clamp_semitone_shift(raw.get("semitone_shift") or 0, conversion_mode),
        "diffusion_steps": clamp_diffusion_steps(raw.get("diffusion_steps") or 0, conversion_mode),
        "vocal_gain_db": clamp_gain_db(raw.get("vocal_gain_db") or 0),
        "source_ext": safe_ext(raw.get("source_ext")),
    }
    if mode == "song":
        params["separation_model"] = check_model(
            raw.get("separation_model") or DEFAULT_SEPARATION_MODEL
        )
    return params


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


@app.function(
    image=api_image,
    volumes={DATA_DIR: data_vol},
    timeout=PIPELINE_TIMEOUT,
    # retries=0: a retry would restart a job whose `failed` status the client
    # has already polled, and `jobs.update` refuses to leave a terminal state
    # anyway. Fail once, visibly.
    retries=0,
)
def run_song_pipeline(job_id: str, params: dict) -> str:
    """separate → convert → mix. Runs in the background; `/status` is the view."""
    from .conversion import VoiceConverter
    from .mixing import mix
    from .separation import INSTRUMENTAL_STEM, VOCAL_STEM, Separator

    data_vol.reload()
    try:
        jobs.update(job_id, jobs.SEPARATING)
        stems = Separator(model=params["separation_model"]).separate.remote(
            storage.get(job_id, INPUT), params["source_ext"]
        )
        storage.put(job_id, VOCAL, stems[VOCAL_STEM])
        storage.put(job_id, INSTRUMENTAL, stems[INSTRUMENTAL_STEM])
        data_vol.commit()

        jobs.update(job_id, jobs.CONVERTING)
        converted = VoiceConverter(mode="singing").convert.remote(
            source_wav=stems[VOCAL_STEM],
            reference_wav=storage.get(job_id, REFERENCE),
            # One shift for the whole track, computed by the caller. Never
            # per chunk, never re-detected here — see the Phase 1 rules.
            semitone_shift=params["semitone_shift"],
            diffusion_steps=params["diffusion_steps"],
        )
        storage.put(job_id, CONVERTED, converted)
        data_vol.commit()

        jobs.update(job_id, jobs.MIXING)
        storage.put(
            job_id,
            OUTPUT,
            mix(converted, stems[INSTRUMENTAL_STEM], vocal_gain_db=params["vocal_gain_db"]),
        )
        data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        raise


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
def run_speech_pipeline(job_id: str, params: dict) -> str:
    """convert → encode. No separation and no mix: `queued → converting → done`."""
    from .conversion import VoiceConverter
    from .mixing import to_mp3

    data_vol.reload()
    try:
        jobs.update(job_id, jobs.CONVERTING)
        converted = VoiceConverter(mode="speech").convert.remote(
            source_wav=storage.get(job_id, INPUT),
            reference_wav=storage.get(job_id, REFERENCE),
            semitone_shift=params["semitone_shift"],
            diffusion_steps=params["diffusion_steps"],
        )
        storage.put(job_id, CONVERTED, converted)
        # The mp3 encode is cheap and needs no separate state; the bar sits at
        # `converting` until the file is on the Volume.
        storage.put(job_id, OUTPUT, to_mp3(converted))
        data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        raise


def spawn(mode: str, job_id: str, params: dict) -> None:
    """Start the branch for `mode` and return without waiting for it."""
    pipeline = run_song_pipeline if jobs.check_mode(mode) == "song" else run_speech_pipeline
    pipeline.spawn(job_id, params)
