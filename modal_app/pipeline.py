"""Orchestration: one job from uploaded bytes to a finished mp3.

    song:    input ──► separate ──► vocal ──► convert ──► mix ──► output.mp3
                            └────► instrumental ───────────┘
    beat:    input ──► separate ──► vocal ──► convert ──────────► mix ──► output.mp3
                            └────► instrumental ──► (đo BPM/key)  ↑
             beat file ────────────────────────────► fit ─────────┘
             hoặc mô tả ──► sinh beat ──────────────► fit ────────┘
    vocal:   input ─────────────────────────► convert ──► encode ──► output.mp3
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

from . import audit, jobs, storage, voices, watermark
from .app import DATA_DIR, api_image, app, data_vol
from .audio_utils import clamp_cfg_rate, clamp_diffusion_steps, clamp_semitone_shift
from .enhance import clamp_clarity
from .mixing import clamp_gain_db
from .prosody import DEFAULT_EMOTION, clamp_expressiveness, clean_emotion
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
# The `beat` branch: the replacement backing track as it arrived (uploaded or
# generated), and the same thing after it has been cut, transposed, stretched
# and looped to fit the song. Both on the Volume so a restarted container does
# not pay for the GPU that made the first one twice.
BEAT = "beat.wav"
BED = "bed.wav"
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

# Which job modes run the separator. Both of them start from a mixed recording
# and need the voice on its own; they differ in what goes back underneath it.
SEPARATING_MODES = ("song", "beat")

# Mirrored from `beatgen.MAX_PROMPT_CHARS`, and duplicated rather than imported
# for the reason every constant in this module is: `pipeline` runs on the API
# image, and `beatgen` imports `stable_audio_tools`.
BEAT_PROMPT_CHARS = 300

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
        # How hard the sampler is pushed towards the reference voice, and how
        # much of `enhance`'s chain runs on the result. Both are `None`-means-
        # default rather than falsy-means-default: 0 is a real setting for each
        # of them and `or` would swallow it.
        "cfg_rate": clamp_cfg_rate(raw.get("cfg_rate")),
        "clarity": clamp_clarity(raw.get("clarity")),
        # A trained voice to convert with, or "" for zero shot. Cleaned rather
        # than checked: whether a profile exists is a question for the GPU
        # container that has the Volume mounted, and an unusable *name* is not
        # worth failing a job over when zero shot is right there.
        "voice_profile": voices.clean_name(raw.get("voice_profile")),
        "source_ext": safe_ext(raw.get("source_ext")),
        # Not a client setting: `WATERMARK` is deployment config, resolved once
        # when the job is created so the record says what was actually done to
        # this file rather than what the config happens to say later.
        "watermark": watermark.enabled(),
    }
    if mode in SEPARATING_MODES:
        params["separation_model"] = check_model(
            raw.get("separation_model") or DEFAULT_SEPARATION_MODEL
        )
    if mode == "beat":
        # Empty means "a beat was uploaded"; `api.submit` is where the two are
        # required to be one or the other, because only it can see the upload.
        params["beat_prompt"] = str(raw.get("beat_prompt") or "").strip()[:BEAT_PROMPT_CHARS]
        # -1 is "a different beat every time", which is what somebody
        # auditioning one wants. A fixed seed is how they get the same one back
        # after changing something else about the job.
        try:
            params["beat_seed"] = int(
                raw.get("beat_seed") if raw.get("beat_seed") is not None else -1
            )
        except (TypeError, ValueError):
            params["beat_seed"] = -1
    if mode == "tts":
        # The language is refused rather than defaulted when it is unknown: a
        # job that silently reads Vietnamese text with the English checkpoint
        # produces a confident recording of nonsense, which is worse than an
        # error. `source_ext` means nothing on this branch — there is no file.
        params["language"] = check_language(raw.get("language") or DEFAULT_LANGUAGE)
        params["speaking_rate"] = clamp_speaking_rate(raw.get("speaking_rate"))
        # How it is read, as opposed to what is read. Unlike the language these
        # two are clamped rather than refused: an unknown style falls back to
        # the natural reading, which is the reading this branch shipped with,
        # and no request is worth failing over a knob that has a safe answer.
        params["emotion"] = clean_emotion(raw.get("emotion") or DEFAULT_EMOTION)
        params["expressiveness"] = clamp_expressiveness(raw.get("expressiveness"))
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
        cfg=params.get("cfg_rate"),
        clarity=params.get("clarity"),
        # The *name* of a trained voice, which is a setting like any other.
        # Never the audio it was trained on, which left the server with the
        # rest of the run's working files.
        profile=params.get("voice_profile") or None,
        model=params.get("separation_model"),
        # Whether a beat was described rather than uploaded — not the words,
        # which are the user's the same way the audio is.
        generated_beat=bool(params.get("beat_prompt")) or None,
        # The language and the style, never the text: what was said is the
        # user's, the same way the audio is (plan §8 item 5). How it was read is
        # a setting, and knowing which styles anybody picks is the only way to
        # find out whether the list is the right list.
        language=params.get("language"),
        emotion=params.get("emotion"),
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
            converted = VoiceConverter(
                mode="singing", voice=params.get("voice_profile", "")
            ).convert.remote(
                source_wav=vocal,
                reference_wav=reference,
                # One shift for the whole track, computed by the caller. Never
                # per chunk, never re-detected here — see the Phase 1 rules.
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
                inference_cfg_rate=params["cfg_rate"],
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
                    # On the vocal alone — `mixing.mix` puts it ahead of the
                    # `amix`, so the backing track is the separator's output
                    # untouched.
                    clarity=params["clarity"],
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
def run_beat_pipeline(job_id: str, params: dict) -> str:
    """separate → (sinh beat) → convert → fit → mix. The backing track is replaced.

    The same first half as `run_song_pipeline` and a different second one: the
    original instrumental is separated out, **measured**, and then thrown away.
    It is the timing reference, not part of the output — the singer's key and
    tempo live in it, and a replacement bed has to match both or it is not a
    backing track, it is a second piece of music playing at the same time.

    Measured on the instrumental rather than on the vocal, deliberately. A
    vocal on its own has almost no pulse to find — that is what the drums were
    for — and a key estimate from one melodic line is a guess. The instrumental
    has both, right up until the moment it is discarded.

    The beat itself arrives one of two ways and the difference is one `if`: a
    file somebody uploaded, or a description handed to a GPU. Everything after
    that point is identical, which is the whole reason `beats.py` takes audio
    and not a source.
    """
    from . import beats
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

        beat = _done_already(job_id, BEAT)
        if beat is None:
            # An uploaded beat is already on the Volume under this name, put
            # there by `_start_job`; only a described one has to be made.
            jobs.update(job_id, jobs.GENERATING)
            from .beatgen import BeatGenerator

            beat = BeatGenerator().generate.remote(
                prompt=params["beat_prompt"], seed=params["beat_seed"]
            )
            storage.put(job_id, BEAT, beat)
            data_vol.commit()

        jobs.update(job_id, jobs.CONVERTING)
        converted = _done_already(job_id, CONVERTED)
        if converted is None:
            reference = storage.get(job_id, REFERENCE)
            # Never auto-detected here, same as `song`: the bed is being fitted
            # to the singer's key, so moving the singer would be fitting two
            # things to each other at once.
            shift = _resolve_shift(job_id, params, vocal, reference, "singing")
            converted = VoiceConverter(
                mode="singing", voice=params.get("voice_profile", "")
            ).convert.remote(
                source_wav=vocal,
                reference_wav=reference,
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
                inference_cfg_rate=params["cfg_rate"],
            )
            storage.put(job_id, CONVERTED, converted)
            data_vol.commit()

        jobs.update(job_id, jobs.MIXING)
        bed = _done_already(job_id, BED)
        if bed is None:
            bed, plan, source, target = beats.analyse_and_fit(beat, instrumental)
            print(f"[beat] {job_id}: beat {source} / song {target} -> {plan}")
            jobs.record_params(job_id, {"beat_fit": str(plan)})
            storage.put(job_id, BED, bed)
            data_vol.commit()

        if not storage.exists(job_id, OUTPUT):
            storage.put(
                job_id,
                OUTPUT,
                mix(
                    converted,
                    bed,
                    vocal_gain_db=params["vocal_gain_db"],
                    watermark=_watermark(job_id, params),
                    clarity=params["clarity"],
                ),
            )
            data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        _finished(job_id, "beat", params, started)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        _finished(job_id, "beat", params, started, exc)
        raise


def _convert_uploaded(job_id: str, params: dict, mode: str) -> str:
    """convert → encode, for the two branches that have nothing to separate.

    `queued → converting → done`, from a file the user uploaded, with no stem
    and no mix. The only thing `speech` and `vocal` disagree about is which
    checkpoint converts them — `jobs.CONVERSION_MODE` answers that — and that
    difference reaches all the way down: the singing checkpoint is F0
    conditioned and runs at 44.1 kHz, and it also means `_resolve_shift` leaves
    the key alone rather than moving the singer into the reference's speaking
    register (`AUTO_DETECT_MODES`).

    Written once rather than twice because everything else here is exactly the
    same code, and the version of this that was copied would have been the
    version where one of the two quietly stopped being watermarked.
    """
    from .conversion import VoiceConverter
    from .mixing import to_mp3

    conversion_mode = jobs.CONVERSION_MODE[mode]
    started = time.monotonic()
    data_vol.reload()
    try:
        jobs.update(job_id, jobs.CONVERTING)
        converted = _done_already(job_id, CONVERTED)
        if converted is None:
            source = storage.get(job_id, INPUT)
            reference = storage.get(job_id, REFERENCE)
            # No separation on these branches, so the input already is the voice.
            shift = _resolve_shift(job_id, params, source, reference, conversion_mode)
            converted = VoiceConverter(
                mode=conversion_mode, voice=params.get("voice_profile", "")
            ).convert.remote(
                source_wav=source,
                reference_wav=reference,
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
                inference_cfg_rate=params["cfg_rate"],
            )
            storage.put(job_id, CONVERTED, converted)
        # The mp3 encode is cheap and needs no separate state; the bar sits at
        # `converting` until the file is on the Volume.
        if not storage.exists(job_id, OUTPUT):
            storage.put(
                job_id,
                OUTPUT,
                to_mp3(
                    converted,
                    watermark=_watermark(job_id, params),
                    clarity=params["clarity"],
                ),
            )
        data_vol.commit()

        jobs.update(job_id, jobs.DONE)
        _finished(job_id, mode, params, started)
        return job_id
    except Exception as exc:
        jobs.fail(job_id, _error_text(exc))
        data_vol.commit()
        _finished(job_id, mode, params, started, exc)
        raise


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
def run_speech_pipeline(job_id: str, params: dict) -> str:
    """A recording of somebody talking, converted with the speech checkpoint."""
    return _convert_uploaded(job_id, params, "speech")


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
def run_vocal_pipeline(job_id: str, params: dict) -> str:
    """A vocal take, converted with the singing checkpoint and nothing else done.

    The `song` branch without the separator: no stem extraction, no instrumental
    to mix back, and the result is the converted voice on its own. For anything
    already dry — an a cappella, a recorded take, a stem somebody else split —
    that is both faster and *better*, because a separator's output carries
    artefacts the converter would otherwise be asked to reproduce as if they
    were part of the singing.

    Handing this branch a full mix is allowed and is not what it is for: the
    backing track goes through Seed-VC along with the voice, which is a thing
    worth hearing once and not a way to convert a song.
    """
    return _convert_uploaded(job_id, params, "vocal")


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
                emotion=params["emotion"],
                expressiveness=params["expressiveness"],
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
            converted = VoiceConverter(
                mode="speech", voice=params.get("voice_profile", "")
            ).convert.remote(
                source_wav=spoken,
                reference_wav=reference,
                semitone_shift=shift,
                diffusion_steps=params["diffusion_steps"],
                inference_cfg_rate=params["cfg_rate"],
            )
            storage.put(job_id, CONVERTED, converted)
        if not storage.exists(job_id, OUTPUT):
            storage.put(
                job_id,
                OUTPUT,
                to_mp3(
                    converted,
                    watermark=_watermark(job_id, params),
                    clarity=params["clarity"],
                ),
            )
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
    "beat": run_beat_pipeline,
    "vocal": run_vocal_pipeline,
    "speech": run_speech_pipeline,
    "tts": run_tts_pipeline,
}


def spawn(mode: str, job_id: str, params: dict) -> None:
    """Start the branch for `mode` and return without waiting for it."""
    PIPELINES[jobs.check_mode(mode)].spawn(job_id, params)
