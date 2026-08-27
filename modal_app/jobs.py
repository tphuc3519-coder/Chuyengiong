"""Job state machine, stored in `modal.Dict`.

    queued → separating → converting → mixing → done
                                             ↘ failed

Transitions may skip forward but never go back: the `speech` branch has no
separation and no mixing, so it runs `queued → converting → done` through the
same machine. Anything that moves backwards is a bug in the pipeline, and this
module raises rather than quietly recording it.

Every function takes an optional `store`, which is why the tests can drive the
whole machine with a plain dict and no Modal credentials. The default is the
`vc-jobs` Dict from `app.py`.

Concurrency note: read-modify-write on a `modal.Dict` is not atomic. One job is
advanced by one pipeline call at a time, so the only writer racing anyone is
the cleanup cron, and it only touches records the TTL already expired.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, MutableMapping
from typing import Any

from .app import job_dict

QUEUED = "queued"
SEPARATING = "separating"
CONVERTING = "converting"
MIXING = "mixing"
DONE = "done"
FAILED = "failed"

# Forward order of the machine. `failed` is not in here: it is reachable from
# any non-terminal state and is compared against separately.
ORDER = (QUEUED, SEPARATING, CONVERTING, MIXING, DONE)
STATUSES = (*ORDER, FAILED)
TERMINAL = frozenset({DONE, FAILED})

# Progress when *entering* a state. The plan quotes two different sets of
# numbers (§4 lists the value each step ends on, §5's pipeline the value it
# starts on); these are §5's, because that is the code that calls `update`.
# They only exist so the bar keeps moving — do not read them as a real estimate.
PROGRESS = {QUEUED: 0, SEPARATING: 5, CONVERTING: 30, MIXING: 75, DONE: 100}

# The user-facing job modes, which are *not* the conversion modes: a `song` job
# separates stems and then converts with the singing checkpoint. Phase 3 maps
# across with this rather than passing "song" into `VoiceConverter`.
JOB_MODES = ("song", "speech")
CONVERSION_MODE = {"song": "singing", "speech": "speech"}


class JobError(RuntimeError):
    """Unknown job, unknown status, or an illegal transition."""


def _store(store: MutableMapping[str, Any] | None) -> MutableMapping[str, Any]:
    return job_dict if store is None else store


def check_mode(mode: str) -> str:
    if mode not in JOB_MODES:
        raise JobError(f"mode must be one of {JOB_MODES}, got {mode!r}")
    return mode


def check_status(status: str) -> str:
    if status not in STATUSES:
        raise JobError(f"status must be one of {STATUSES}, got {status!r}")
    return status


def check_transition(current: str, new: str) -> str:
    """Raise unless `current → new` is a legal move.

    Terminal is the only rule left. A running job used to be forbidden from
    re-entering a state it had already been in, on the reasoning that a machine
    only moves forward — but Modal restarts a preempted container with the same
    input, and a restarted `run_song_pipeline` starts again from the top. It
    genuinely is separating again. Refusing to say so killed the job outright,
    over a routine bit of infrastructure, with the memorable line

        JobError: cannot move backwards: separating → separating

    Progress does not follow the status back: `update` keeps it monotonic, so a
    rewind redoes the work without the bar reading as a crash.
    """
    check_status(new)
    if current in TERMINAL:
        raise JobError(f"job is already {current}, cannot move to {new}")
    return new


# --- records --------------------------------------------------------------


def create(
    job_id: str,
    mode: str,
    params: dict | None = None,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
) -> dict:
    """Register a new job in `queued`. The caller owns id generation."""
    store = _store(store)
    if job_id in store:
        raise JobError(f"job {job_id} already exists")
    stamp = time.time() if now is None else now
    record = {
        "id": job_id,
        "status": QUEUED,
        "progress": PROGRESS[QUEUED],
        "mode": check_mode(mode),
        "created_at": stamp,
        "updated_at": stamp,
        "error": None,
        "params": dict(params or {}),
    }
    store[job_id] = record
    return record


def find(job_id: str, store: MutableMapping[str, Any] | None = None) -> dict | None:
    """The record, or None. Use this on the read path; `get` on the write path."""
    record = _store(store).get(job_id)
    # A copy, deliberately: callers mutate what they get back, and on a
    # `modal.Dict` a mutation only lands when it is written back.
    return dict(record) if record is not None else None


def get(job_id: str, store: MutableMapping[str, Any] | None = None) -> dict:
    record = find(job_id, store)
    if record is None:
        raise JobError(f"no such job: {job_id}")
    return record


def update(
    job_id: str,
    status: str | None = None,
    progress: int | None = None,
    error: str | None = None,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
) -> dict:
    """Advance a job. `status` alone sets the progress that goes with it.

    Progress never decreases, whatever it is asked to do — a bar that walks
    backwards reads as a crash to the user even when the job is healthy.
    """
    store = _store(store)
    record = get(job_id, store)

    if status is not None:
        check_transition(record["status"], status)
        record["status"] = status
        if progress is None:
            progress = PROGRESS.get(status, record["progress"])
        if status == FAILED:
            record["error"] = error or "unknown error"

    if error is not None:
        record["error"] = error
    if progress is not None:
        record["progress"] = max(record["progress"], min(100, max(0, int(progress))))

    record["updated_at"] = time.time() if now is None else now
    store[job_id] = record
    return record


def fail(job_id: str, error: str, store: MutableMapping[str, Any] | None = None) -> dict:
    """Terminal failure. Never raises on an already-failed job: the pipeline's
    `except` block must not itself explode while reporting an error."""
    record = get(job_id, store)
    if record["status"] in TERMINAL:
        return record
    return update(job_id, status=FAILED, error=error, store=store)


def record_params(
    job_id: str,
    updates: dict,
    store: MutableMapping[str, Any] | None = None,
) -> dict:
    """Write back parameters the pipeline worked out for itself.

    Phase 5 is the only caller: `semitone_shift` arrives as None when the
    client asked for auto-detect, and the value measured off the vocal stem has
    to land somewhere the status endpoint can read it. Anything else about a
    job is fixed when it is created.
    """
    store = _store(store)
    record = get(job_id, store)
    record["params"] = {**record.get("params", {}), **updates}
    store[job_id] = record
    return record


def public(record: dict) -> dict:
    """What `/status/{id}` returns. `params` stays server-side — except one.

    `semitone_shift` is the exception because the pipeline may have chosen it
    rather than the client (plan §7): a user who asked for auto-detect has no
    other way to see what was applied, and no way to decide what to override it
    with next time. It reads None until separation is done and the measurement
    has run.
    """
    return {
        "id": record["id"],
        "status": record["status"],
        "progress": record["progress"],
        "mode": record["mode"],
        "error": record["error"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "semitone_shift": record.get("params", {}).get("semitone_shift"),
    }


# --- expiry ---------------------------------------------------------------


def expired_ids(
    max_age_hours: float,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
) -> list[str]:
    """Ids of records older than the TTL, by `created_at`."""
    store = _store(store)
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    stale = []
    for job_id in list(store.keys()):
        record = store.get(job_id)
        if isinstance(record, dict) and record.get("created_at", 0) <= cutoff:
            stale.append(job_id)
    return stale


def forget(job_ids: Iterable[str], store: MutableMapping[str, Any] | None = None) -> int:
    """Drop records. Returns how many were actually there."""
    store = _store(store)
    dropped = 0
    for job_id in job_ids:
        if job_id not in store:
            continue
        try:
            del store[job_id]
        except KeyError:  # raced another sweep between the check and the delete
            continue
        dropped += 1
    return dropped
