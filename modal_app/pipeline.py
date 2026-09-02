"""Orchestration: one job from uploaded bytes to a finished mp3.

    song:    input ──► separate ──► vocal ──► convert ──► mix ──► output.mp3
                            └────► instrumental ───────────┘
    speech:  input ─────────────────────────► convert ──► encode ──► output.mp3
    tts:     input.txt ──► synthesize ──────► convert ──► encode ──► output.mp3

These functions run on the small CPU image and do no audio work themselves:
separation and conversion happen in their own GPU containers, called with
`.remote()` so this one blocks on them without holding a GPU. `/submit` starts
a job with `.spawn()` and returns immediately.

The job's parameters are normalised **once**, in `clean_params`, before the job
record is written. Every clamp in the plan (semitone range per mode, diffusion
step range, vocal gain) is applied there rather than at the point of use, so
what the status endpoint reports and what the GPU receives cannot disagree.

One parameter is not known at that point: `semitone_shift` is None when the
client asked for auto-detect, and the measurement it needs runs here, on the
whole vocal stem, after separation and before chunking (plan §7). `_resolve_shift`
is the single place that happens, and it writes the answer back into the job
record so `/status` can show what was applied.
"""

from __future__ import annotations

import time

from . import audit, jobs, storage, watermark
from .app import DATA_DIR, api_image, app, data_vol
from .audio_utils import clamp_diffusion_steps, clamp_semitone_shift
from .mixing import clamp_gain_db
from .separation import DEFAULT_SEPARATION_MODEL, check_model, safe_ext
from .tts import DEFAULT_LANGUAGE, check_language, clamp_speaking_rate

# Artifact names on the Volume, matching the layout in `storage`. `input.mp3`
# is a label, not a claim about the container: the upload keeps whatever format
# it arrived in and `params["source_ext"]` carries the real extension through to
# the separator, which picks its decoder by file name.
INPUT = "input.mp3"
# The `tts` branch's input, in place of `input.mp3`: UTF-8 text, on the Volume
# rather than in the job record so the TTL sweep expires it with the audio.
TEXT = "input.txt"
REFERENCE = "reference.wav"
VOCAL = "vocal.wav"
INSTRUMENTAL = "instrumental.wav"
# What the synthesiser produced, before it has anybody's voice.
SPOKEN = "spoken.wav"
CONVERTED = "converted.wav"
OUTPUT = "output.mp3"

# A 15 minute song at 44.1kHz is ~10 minutes of GPU work in the worst case;
# 30 minutes of wall clock leaves room for a cold start on both models.
PIPELINE_TIMEOUT = 1800

# Which modes measure a shift when the client does not name one.
#
# Not `singing`, and this is the whole reason the tuple exists. A song's vocal
# is converted, shifted, and then mixed back over an instrumental that nothing
# in this pipeline transposes — so any shift that is not a whole octave leaves
# the singer in a different key from the backing track, which is what "lệch
# tone" means and it is not subtle. The measurement makes it worse: it compares
# the singer's median F0 against the reference, and the reference is somebody
# *speaking*, so the distance between them is a musically arbitrary number that
# lands on a multiple of twelve only by accident.
#
# Speech has neither problem — nothing is mixed under it, and moving a talker
# into the target's natural range is the point. So it keeps auto-detect, and a
# song keeps the key it was written in. The slider still moves either way, for
# anyone who wants an octave.
#
# These are conversion modes, not job modes, which is why `tts` is not listed
# and still gets measured: it converts as `speech`, and it needs the
# measurement more than an upload does — the synthetic voice has one register
# per language and the target is as likely to be an octave off it as not.
AUTO_DETECT_MODES = ("speech",)

# Job records are read by a browser that is polling; keep failure text short
# enough to render and free of stack traces.
MAX_ERROR_CHARS = 400

# Between `mixing` (75) and `done` (100). Not a status of its own: watermarking
# is part of producing the output file, and adding a sixth state would mean a
# new label in the UI for a step the user has no decision to make about.
WATERMARK_PROGRESS = 85


def clean_params(mode: str, raw: dict | None = None) -> dict:
    """Validate and clamp everything `/submit` accepts. Pure, so tests cover it.

    Raises `jobs.JobError` for an unusable mode, `SeparationError` for an
    unknown separation model and `TtsError` for a language we do not speak;
    every other field is clamped rather than rejected, because a slider that
    arrives out of range is a client bug, not something worth failing a paid
    GPU job over.
    """
    raw = dict(raw or {})
    jobs.check_mode(mode)
    conversion_mode = jobs.CONVERSION_MODE[mode]

    # None is a value here, not a missing one: it means "work it out from the
    # audio" (plan §7). `or` would flatten it into 0, which is a real setting.
    shift = raw.get("semitone_shift")
    params = {
        "semitone_shift": None if shift is None else clamp_semitone_shift(shift, conversion_mode),
        "diffusion_steps": clamp_diffusion_steps(raw.get("diffusion_steps") or 0, conversion_mode),
        "vocal_gain_db": clamp_gain_db(raw.get("vocal_gain_db") or 0),
        "source_ext": safe_ext(raw.get("source_ext")),
        # Not a client setting: `WATERMARK` is deployment config, resolved once
        # when the job is created so the record says what was actually done to
        # this file rather than what the config happens to say later.
        "watermark": watermark.enabled(),
    }
    if mode == "song":
        params["separation_model"] = check_model(
            raw.get("separation_model") or DEFAULT_SEPARATION_MODEL
        )
    if mode == "tts":
        # The language is refused rather than defaulted when it is unknown: a
        # job that silently reads Vietnamese text with the English checkpoint
        # produces a confident recording of nonsense, which is worse than an
        # error. `source_ext` means nothing on this branch — there is no file.
        params["language"] = check_language(raw.get("language") or DEFAULT_LANGUAGE)
        params["speaking_rate"] = clamp_speaking_rate(raw.get("speaking_rate"))
        params.pop("source_ext", None)
    return params


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


def _finished(
    job_id: str,
    mode: str,
    params: dict,
    started: float,
    exc: BaseException | None = None,
) -> None:
    """Close the audit trail for one run (plan §8 item 5).

    On failure this logs the exception's *class*, not `_error_text`: that
    message can quote ffmpeg on a file the user uploaded, and the status
    endpoint already carries it to the one person entitled to read it.
    """
    audit.record(
        audit.FAILED if exc is not None else audit.DONE,
        job_id,
        mode=mode,
        seconds=time.monotonic() - started,
        shift=params.get("semitone_shift"),
        steps=params.get("diffusion_steps"),
        model=params.get("separation_model"),
        # The language, never the text: what was said is the user's, the same
        # way the audio is (plan §8 item 5).
        language=params.get("language"),
        watermark=params.get("watermark"),
        reason=type(exc).__name__ if exc is not None else None,
    )


def _watermark(job_id: str, params: dict):
    """The callable `mixing` applies between the mix and the encode, or None.

    It bumps progress on the way past. Watermarking a long song is a minute of
    CPU inside the `mixing` step, and plan §6 asks for a bar that never sits
    still for 20 seconds; this is the only moment where the step's own
    boundaries do not provide one.
    """
    if not params.get("watermark"):
        return None

    from .watermark import Watermarker

    def apply(audio_wav: bytes) -> bytes:
        jobs.update(job_id, progress=WATERMARK_PROGRESS)
        return Watermarker().embed.remote(audio_wav)

    return apply


def _resolve_shift(job_id: str, params: dict, source: bytes, reference: bytes, mode: str) -> int:
    """The one place a pitch shift is decided, and the only time it is measured.

    Returns what the client asked for when it asked for something. Otherwise it
    measures the median F0 of both sides — on the **whole** source, before any
    chunking — and records the answer so `/status` can report it.

    Everything about this is once-per-job on purpose. A shift computed per
    chunk drifts across a song, which plan §10 lists as the most common bug in
    this kind of app, and the reason `convert()` takes the value as an argument
    instead of detecting it.
    """
    if params["semitone_shift"] is not None:
        return params["semitone_shift"]

    if mode not in AUTO_DETECT_MODES:
        params["semitone_shift"] = 0
        jobs.record_params(job_id, {"semitone_shift": 0})
        return 0

    from .pitch import suggest_semitone_shift

    shift = clamp_semitone_shift(suggest_semitone_shift(source, reference, mode), mode)
    params["semitone_shift"] = shift
    jobs.record_params(job_id, {"semitone_shift": shift})
    return shift


def _done_already(job_id: str, name: str) -> bytes | None:
    """What an earlier run of this job already finished, or `None`.

    Modal restarts a preempted container with the same input, and the restart
    begins at the top of the pipeline — so a song that was preempted while
    converting would separate itself all over again, on a GPU, for nothing. Each
    stage asks this first instead.

    Safe because `storage.put` writes to a temp name and renames: an artifact
    that is there is whole, and a stage killed mid-write leaves nothing to
    mistake for a finished one.
    """
    return storage.get(job_id, name) if storage.exists(job_id, name) else None


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

    started = time.monotonic()
    data_vol.reload()
    try:
        jobs.update(job_id, jobs.SEPARATING)
        vocal = _done_already(job_id, VOCAL)
        instrumental = _done_already(job_id, INSTRUMENTAL)
        if vocal is None or instrumental is None:
            stems = Separator(model=params["separation_model"]).separate.remote(
                storage.get(job_id, INPUT), params["source_ext"]
            )
            vocal, instrumental = stems[VOCAL_STEM], stems[INSTRUMENTAL_STEM]
            storage.put(job_id, VOCAL, vocal)
            storage.put(job_id, INSTRUMENTAL, instrumental)
            data_vol.commit()

        jobs.update(job_id, jobs.CONVERTING)
        converted = _done_already(job_id, CONVERTED)
        if converted is None:
            reference = storage.get(job_id, REFERENCE)
            # After separation, before chunking (plan §7). The vocal stem is what
            # gets measured — silence detection and F0 on a full mix would both be
            # reading the backing track as well as the singer.
            shift = _resolve_shift(job_id, params, vocal, reference, "singing")
            converted = VoiceConverter(mode="singing").convert.remote(
                source_wav=vocal,
                reference_wav=reference,
                # One shift for the whole track, computed by the caller. Never
                # per chunk, never re-detected here — see the Phase 1 rules.
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
            )
            storage.put(job_id, CONVERTED, converted)
            data_vol.commit()

        jobs.update(job_id, jobs.MIXING)
        if not storage.exists(job_id, OUTPUT):
            storage.put(
                job_id,
                OUTPUT,
                mix(
                    converted,
                    instrumental,
                    vocal_gain_db=params["vocal_gain_db"],
                    watermark=_watermark(job_id, params),
                ),
            )
            data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        _finished(job_id, "song", params, started)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        _finished(job_id, "song", params, started, exc)
        raise


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
def run_speech_pipeline(job_id: str, params: dict) -> str:
    """convert → encode. No separation and no mix: `queued → converting → done`."""
    from .conversion import VoiceConverter
    from .mixing import to_mp3

    started = time.monotonic()
    data_vol.reload()
    try:
        jobs.update(job_id, jobs.CONVERTING)
        converted = _done_already(job_id, CONVERTED)
        if converted is None:
            source = storage.get(job_id, INPUT)
            reference = storage.get(job_id, REFERENCE)
            # No separation on this branch, so the input already is the voice.
            shift = _resolve_shift(job_id, params, source, reference, "speech")
            converted = VoiceConverter(mode="speech").convert.remote(
                source_wav=source,
                reference_wav=reference,
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
            )
            storage.put(job_id, CONVERTED, converted)
        # The mp3 encode is cheap and needs no separate state; the bar sits at
        # `converting` until the file is on the Volume.
        if not storage.exists(job_id, OUTPUT):
            storage.put(job_id, OUTPUT, to_mp3(converted, watermark=_watermark(job_id, params)))
        data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        _finished(job_id, "speech", params, started)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        _finished(job_id, "speech", params, started, exc)
        raise


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
def run_tts_pipeline(job_id: str, params: dict) -> str:
    """synthesize → convert → encode. `queued → synthesizing → converting → done`.

    The second half is `run_speech_pipeline` verbatim, and deliberately so: once
    the text is a wav there is nothing about it that makes it different from a
    recording somebody uploaded, and the conversion, the pitch measurement, the
    level and the watermark are all better for being the same code.
    """
    from .conversion import VoiceConverter
    from .mixing import to_mp3
    from .tts import synthesize

    started = time.monotonic()
    data_vol.reload()
    try:
        jobs.update(job_id, jobs.SYNTHESIZING)
        spoken = _done_already(job_id, SPOKEN)
        if spoken is None:
            text = storage.get(job_id, TEXT).decode("utf-8")
            # Which model reads it is `tts`'s business, not this one's: Japanese
            # goes to a different engine than Vietnamese and the pipeline is the
            # same either way.
            spoken = synthesize(
                params["language"],
                text=text,
                speaking_rate=params["speaking_rate"],
            )
            storage.put(job_id, SPOKEN, spoken)
            data_vol.commit()

        jobs.update(job_id, jobs.CONVERTING)
        converted = _done_already(job_id, CONVERTED)
        if converted is None:
            reference = storage.get(job_id, REFERENCE)
            # Measured on the synthesised voice, exactly as the speech branch
            # measures an uploaded one: the MMS speaker has one register per
            # language and it is nobody's in particular.
            shift = _resolve_shift(job_id, params, spoken, reference, "speech")
            converted = VoiceConverter(mode="speech").convert.remote(
                source_wav=spoken,
                reference_wav=reference,
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
            )
            storage.put(job_id, CONVERTED, converted)
        if not storage.exists(job_id, OUTPUT):
            storage.put(job_id, OUTPUT, to_mp3(converted, watermark=_watermark(job_id, params)))
        data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        _finished(job_id, "tts", params, started)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        _finished(job_id, "tts", params, started, exc)
        raise


# One branch per job mode. A dict rather than a chain of conditionals because
# `jobs.JOB_MODES` and this have to stay the same length, and a missing key is
# a KeyError here instead of a job quietly running the wrong pipeline.
PIPELINES = {
    "song": run_song_pipeline,
    "speech": run_speech_pipeline,
    "tts": run_tts_pipeline,
}


def spawn(mode: str, job_id: str, params: dict) -> None:
    """Start the branch for `mode` and return without waiting for it."""
    PIPELINES[jobs.check_mode(mode)].spawn(job_id, params)
