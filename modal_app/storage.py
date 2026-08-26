"""User files on the Modal Volume, and the cron that expires them.

Layout, exactly as the plan specifies::

    /data/{job_id}/{input,reference,vocal,instrumental,converted,output}.{wav,mp3}

Every function takes a `root` so the tests can run against a tmp directory; the
default is the mounted Volume. Nothing here imports torch, numpy or the audio
stack — the cleanup cron runs on the small image and must stay cheap.

Two rules this module enforces rather than trusts its callers on:

* **Job ids and file names are validated, not interpolated.** `job_id` reaches
  us straight out of a URL path segment (`/status/{id}`, `/download/{id}`), so
  `..` has to be impossible rather than unlikely.
* **Writes are atomic.** A reader in another container must never observe a
  half-written `output.mp3`; we write `.part` and rename.

Round-trip check against the real Volume (needs Modal credentials, no GPU):

    modal run -m modal_app.verify
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import modal

from .app import DATA_DIR, api_image, app, data_vol

DATA_ROOT = Path(DATA_DIR)

# The plan's TTL. Jobs are disposable: the user downloads within minutes or
# converts again.
DEFAULT_MAX_AGE_HOURS = 6
# How often the sweep runs. Equal to the TTL, so nothing lives past ~2x it.
CLEANUP_PERIOD_HOURS = 6

# uuid4().hex — no dashes, so nothing in a job id can ever be a path separator.
_JOB_ID_RE = re.compile(r"[0-9a-f]{32}")
# The artifact names the pipeline writes, and nothing else.
_NAME_RE = re.compile(r"[a-z0-9_]+\.(wav|mp3|json)")


class StorageError(ValueError):
    """Bad job id, bad file name, or a file that is not there."""


def new_job_id() -> str:
    return uuid.uuid4().hex


def check_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise StorageError(f"not a job id: {job_id!r}")
    return job_id


def check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise StorageError(f"not a valid artifact name: {name!r}")
    return name


def _root(root: Path | str | None) -> Path:
    return DATA_ROOT if root is None else Path(root)


def job_dir(job_id: str, root: Path | str | None = None) -> Path:
    return _root(root) / check_job_id(job_id)


def path_for(job_id: str, name: str, root: Path | str | None = None) -> Path:
    return job_dir(job_id, root) / check_name(name)


# --- read / write ---------------------------------------------------------


def put(job_id: str, name: str, data: bytes, root: Path | str | None = None) -> str:
    """Write one artifact. Returns its path."""
    path = path_for(job_id, name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)  # rename is atomic; a reader sees all of it or none of it
    return str(path)


def get(job_id: str, name: str, root: Path | str | None = None) -> bytes:
    path = path_for(job_id, name, root)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise StorageError(f"job {job_id} has no {name}") from exc


def exists(job_id: str, name: str, root: Path | str | None = None) -> bool:
    return path_for(job_id, name, root).is_file()


def size(job_id: str, name: str, root: Path | str | None = None) -> int:
    path = path_for(job_id, name, root)
    try:
        return path.stat().st_size
    except FileNotFoundError as exc:
        raise StorageError(f"job {job_id} has no {name}") from exc


def delete_job(job_id: str, root: Path | str | None = None) -> bool:
    """Remove a whole job directory. False if there was nothing to remove."""
    path = job_dir(job_id, root)
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


# --- expiry ---------------------------------------------------------------


def _newest_mtime(path: Path) -> float:
    """Most recent mtime in a job directory, the directory itself included.

    Age is measured from the last write rather than from creation on purpose:
    a pipeline that is still writing stems keeps its own directory alive, so a
    long job can never have the file it is about to read deleted underneath it.
    """
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:  # vanished mid-scan; another cleanup, or a live job
            continue
    return newest


def job_age_sec(job_id: str, root: Path | str | None = None, now: float | None = None) -> float:
    path = job_dir(job_id, root)
    if not path.is_dir():
        raise StorageError(f"no such job: {job_id}")
    return (time.time() if now is None else now) - _newest_mtime(path)


def backdate(job_id: str, seconds: float, root: Path | str | None = None) -> None:
    """Age a job directory, so the TTL can be exercised without waiting for it."""
    when = time.time() - seconds
    path = job_dir(job_id, root)
    for target in [path, *path.rglob("*")]:
        os.utime(target, (when, when))


def cleanup_expired(
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    root: Path | str | None = None,
    now: float | None = None,
) -> list[str]:
    """Delete job directories untouched for `max_age_hours`.

    Returns the ids removed rather than the plan's bare count: the cron has to
    drop the matching records from the job Dict too, and it needs the ids to do
    that. `len()` gives you the count the plan asked for.

    Anything in the data root that is not a job directory is left alone — this
    runs unattended on a shared Volume, so it only ever removes what it can
    positively identify as ours.
    """
    base = _root(root)
    if not base.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    removed = []
    for path in sorted(base.iterdir()):
        if not path.is_dir() or not _JOB_ID_RE.fullmatch(path.name):
            continue
        if _newest_mtime(path) <= cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


# --- Modal surface --------------------------------------------------------


@app.function(
    image=api_image,
    schedule=modal.Period(hours=CLEANUP_PERIOD_HOURS),
    volumes={DATA_DIR: data_vol},
    timeout=600,
)
def cleanup(max_age_hours: float = DEFAULT_MAX_AGE_HOURS) -> int:
    """Scheduled sweep: expired files off the Volume, expired records out of the Dict."""
    from . import audit, jobs, ratelimit

    data_vol.reload()
    removed = cleanup_expired(max_age_hours)
    # Records can outlive their files (a job that failed before writing
    # anything), so expire those by `created_at` as well.
    stale: Iterable[str] = set(removed) | set(jobs.expired_ids(max_age_hours))
    forgotten = jobs.forget(stale)
    # Rate limit windows are an hour wide and this runs every six, so by now
    # every key still in there is either active or dead weight.
    windows = ratelimit.prune()
    data_vol.commit()
    # The sweep is the other half of the audit trail (plan §8 item 5): a job id
    # that stops appearing in the logs did so because its files were deleted on
    # schedule, and this line is what says so.
    audit.record(audit.EXPIRE, jobs=len(removed), records=forgotten, windows=windows)
    return len(removed)
