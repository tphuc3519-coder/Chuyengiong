"""FastAPI web endpoints, served from Modal as a single ASGI app.

    GET  /health
    POST /submit            multipart: input | text, reference, mode, params, consent
    GET  /status/{job_id}
    GET  /download/{job_id}

Deploy through `modal_app.deploy`, not this module — see its docstring.

Four things this layer owns, and nothing else does:

* **Size limits.** Uploads are read in chunks and cut off at the cap, so a
  client cannot make the container buffer an arbitrary file to find out it is
  too big. Duration limits belong further down, where the audio is decoded, and
  the `tts` branch's text length belongs to `tts.check_text`.
* **The consent gate.** A checkbox the frontend enforces on its own is not a
  gate. `/submit` refuses without it (plan §8, item 1).
* **The rate limit.** Same reason: the browser uploads straight here, so a cap
  enforced in the frontend's route handlers would not be on the upload path at
  all (plan §9, and see `ratelimit`).
* **Volume freshness.** The pipeline writes `output.mp3` in a different
  container, so `/download` reloads the Volume before it looks.
* **The audit trail.** A submit and a download are the two moments a person is
  on the other end of the wire; both are logged as a job id, a timestamp and
  nothing about the audio (plan §8 item 5, and see `audit`).
"""

import os
from typing import Annotated

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import audit, jobs, pipeline, ratelimit, storage
from .app import APP_NAME, DATA_DIR, api_image, app, config_secret, data_vol
from .audio_utils import AudioError
from .separation import DEFAULT_SEPARATION_MODEL, SeparationError, safe_ext
from .tts import DEFAULT_LANGUAGE, DEFAULT_SPEAKING_RATE, TtsError, check_text

# 60 MB is ~40 minutes of 192k mp3 — well past the 15 minute duration limit, so
# anything larger is a mistake rather than a long song.
MAX_INPUT_BYTES = 60 * 1024 * 1024
# The reference is 5–30 seconds of speech. 20 MB is generous even for wav.
MAX_REFERENCE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024

web = FastAPI(title="voice-convert API", version="0.4.0")


def allowed_origins() -> list[str]:
    """Origins the browser may call from. `*` until a domain is configured.

    Which origins are allowed is deployment config, not code: set
    ALLOWED_ORIGINS (comma separated) on the Modal secret once the Vercel
    domain exists. Nothing here is authenticated, so a permissive default costs
    nothing beyond the rate limit that is already on `/submit`.
    """
    configured = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")]
    return [o for o in configured if o] or ["*"]


# The browser uploads straight to Modal: a 3 minute mp3 is 4–7 MB and Vercel's
# request body limit is 4.5 MB, so the upload path cannot go through the
# frontend's own API routes (plan §6).
web.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@web.get("/health")
async def health() -> dict:
    """Liveness probe. Deliberately touches no Volume and no GPU."""
    return {"status": "ok", "app": APP_NAME}


async def _read_upload(upload: UploadFile, limit: int, label: str) -> bytes:
    """Read an upload, refusing to buffer more than `limit` bytes."""
    parts, total = [], 0
    while chunk := await upload.read(UPLOAD_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{label} is larger than {limit // (1024 * 1024)} MB")
        parts.append(chunk)
    if not total:
        raise HTTPException(400, f"{label} is empty")
    return b"".join(parts)


def _check_quota(client: str) -> None:
    """Reject early if this client is already at its hourly cap.

    Read-only on purpose: the slot is not spent until the job actually starts
    (`_start_job`), so a request that fails validation does not cost one.
    """
    wait = ratelimit.retry_after(client)
    if not wait:
        return
    raise HTTPException(
        429,
        f"rate limit reached: {ratelimit.max_jobs()} jobs per hour, "
        f"try again in {wait // 60 + 1} minute(s)",
        headers={"Retry-After": str(wait)},
    )


def _start_job(mode: str, params: dict, source: bytes, reference: bytes, client: str) -> str:
    """Persist the uploads, register the job, hand it to the pipeline.

    Files first, record second, spawn last: every state in between is one the
    pipeline can survive, whereas a job that starts before its input is on the
    Volume cannot.

    `source` is the uploaded audio, or on the `tts` branch the UTF-8 text. It
    goes to the Volume either way, and for the same reason: the sweep that
    expires a job's audio after six hours has to take what the user wrote with
    it, and a job record is not where that belongs.
    """
    job_id = storage.new_job_id()
    storage.put(job_id, pipeline.TEXT if mode == "tts" else pipeline.INPUT, source)
    storage.put(job_id, pipeline.REFERENCE, reference)
    data_vol.commit()

    jobs.create(job_id, mode, params=params)
    try:
        # Spend the slot here rather than at the top of the request: between
        # `_check_quota` and this line a parallel upload could take the last
        # one, and letting that request through is cheaper than making every
        # rejected upload cost a slot.
        ratelimit.check(client)
        pipeline.spawn(mode, job_id, params)
    except ratelimit.RateLimited as exc:
        jobs.fail(job_id, "rate limit reached")
        raise HTTPException(429, str(exc), headers={"Retry-After": str(exc.retry_after)}) from exc
    except Exception as exc:  # queue rejected it — say so now, don't leave it queued
        jobs.fail(job_id, f"could not start: {type(exc).__name__}")
        raise HTTPException(503, "could not start the job, try again") from exc
    return job_id


@web.post("/submit")
async def submit(
    request: Request,
    reference: Annotated[UploadFile, File()],
    # Optional because `tts` has no file to send: that branch reads `text`
    # instead, and which of the two is required is decided by `mode` below
    # rather than by the signature.
    source: Annotated[UploadFile | None, File(alias="input")] = None,
    text: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = DEFAULT_LANGUAGE,
    speaking_rate: Annotated[float, Form()] = DEFAULT_SPEAKING_RATE,
    mode: Annotated[str, Form()] = "song",
    # Absent means auto-detect (plan §7), which is not the same as 0 — that is
    # a client explicitly asking for no shift. The pipeline measures the vocal
    # stem when this is None and reports the result through `/status`.
    semitone_shift: Annotated[int | None, Form()] = None,
    diffusion_steps: Annotated[int, Form()] = 0,
    vocal_gain_db: Annotated[float, Form()] = 0.0,
    separation_model: Annotated[str, Form()] = DEFAULT_SEPARATION_MODEL,
    consent: Annotated[bool, Form()] = False,
) -> dict:
    """Start a job. Returns as soon as it is queued.

    `mode` decides what the job starts from: `song` and `speech` take the
    uploaded `input` file, `tts` takes `text` and reads it out in the reference
    voice. `reference` is required either way — it is the voice, which is the
    whole product.
    """
    client = ratelimit.client_key(
        ratelimit.address_from_headers(
            request.headers, request.client.host if request.client else None
        )
    )
    _check_quota(client)
    if not consent:
        raise HTTPException(
            400,
            "consent is required: confirm you have the right to use this voice, "
            "or that it is your own",
        )
    try:
        params = pipeline.clean_params(
            mode,
            {
                "semitone_shift": semitone_shift,
                "diffusion_steps": diffusion_steps,
                "vocal_gain_db": vocal_gain_db,
                "separation_model": separation_model,
                "source_ext": safe_ext(source.filename if source else None),
                "language": language,
                "speaking_rate": speaking_rate,
            },
        )
        # Text and audio are the same thing to everything downstream — the
        # bytes a job starts from — so they are validated in the same place and
        # answered with the same 400.
        text_bytes = check_text(text).encode("utf-8") if mode == "tts" else None
    except (jobs.JobError, SeparationError, AudioError, TtsError, ValueError) as exc:
        # Every one of these is something the client sent, not a fault here.
        raise HTTPException(400, str(exc)) from exc

    if text_bytes is not None:
        source_bytes = text_bytes
    elif source is None:
        raise HTTPException(400, "input file is required for this mode")
    else:
        source_bytes = await _read_upload(source, MAX_INPUT_BYTES, "input")
    reference_bytes = await _read_upload(reference, MAX_REFERENCE_BYTES, "reference")

    job_id = _start_job(mode, params, source_bytes, reference_bytes, client)
    # The audit trail proper (plan §8 item 5): who asked, when, for what shape
    # of job, and that the gate above was passed — no file names, no audio.
    audit.record(
        audit.SUBMIT,
        job_id,
        mode=mode,
        client=client,
        consent=consent,
        # On the `tts` branch this is how long the text was, never what it
        # said — the same rule the audio has been under all along.
        input_bytes=len(source_bytes),
        reference_bytes=len(reference_bytes),
        language=params.get("language"),
    )
    return {
        "job_id": job_id,
        "status": jobs.QUEUED,
        "mode": mode,
        # So the UI can warn before the user hits the wall rather than after.
        "jobs_remaining": ratelimit.remaining(client),
    }


@web.get("/status/{job_id}")
async def status(job_id: str) -> dict:
    """Poll this every 2 seconds. Reads the Dict only — no Volume, no GPU.

    The two 404s say different things on purpose. A job id is 32 hex
    characters, and the place people read one off is a console that truncates
    long strings with an ellipsis — so "you did not send me a job id" is a
    likely answer here and a completely different problem from "that job is
    gone". Both used to be "no such job", which sent someone hunting for a
    missing job when they had only pasted half an id.
    """
    try:
        storage.check_job_id(job_id)
    except storage.StorageError as exc:
        raise HTTPException(
            404, f"not a job id: expected 32 hex characters, got {len(job_id)}"
        ) from exc
    record = jobs.find(job_id)
    if record is None:
        raise HTTPException(404, "no such job")
    return jobs.public(record)


@web.get("/download/{job_id}")
async def download(job_id: str) -> Response:
    """The finished mp3. 409 while it is still running, 410 once it expired."""
    record = await status(job_id)
    if record["status"] == jobs.FAILED:
        raise HTTPException(409, record["error"] or "job failed")
    if record["status"] != jobs.DONE:
        raise HTTPException(409, f"job is {record['status']}, not done yet")

    # The pipeline wrote the file in another container; without this reload we
    # would be looking at this container's stale view of the Volume.
    data_vol.reload()
    try:
        data = storage.get(job_id, pipeline.OUTPUT)
    except storage.StorageError as exc:
        raise HTTPException(410, "output has expired, please convert again") from exc

    audit.record(audit.DOWNLOAD, job_id, output_bytes=len(data))
    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="voice-convert-{job_id[:8]}.mp3"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@app.function(
    image=api_image,
    volumes={DATA_DIR: data_vol},
    # CORS is configured while this module is imported, so the values have to
    # be in the environment before that — which is what a secret does.
    secrets=[config_secret()],
    timeout=600,
)
@modal.concurrent(max_inputs=50)  # polling is cheap; do not start a container per poll
@modal.asgi_app()
def api():
    return web
