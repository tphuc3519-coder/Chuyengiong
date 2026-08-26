"""Phase 2 acceptance, run against real Modal infrastructure.

    modal run -m modal_app.verify

The unit tests drive `storage` against a tmp directory and `jobs` against a
plain dict, which proves the logic but proves nothing about the two pieces of
infrastructure underneath it: that a Volume write in one container is readable
after a reload, and that `modal.Dict` really behaves like the mapping `jobs`
assumes (`in`, `.get`, `.keys()`, `del`). Those are exactly the assumptions
that fail in production and cannot fail in CI, so they get their own check.

Needs Modal credentials. No GPU, and it costs a few CPU-seconds.
"""

from __future__ import annotations

from .app import DATA_DIR, api_image, app, data_vol
from .storage import (
    DEFAULT_MAX_AGE_HOURS,
    backdate,
    cleanup_expired,
    delete_job,
    exists,
    get,
    new_job_id,
    put,
)


@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=600)
def check_storage() -> dict:
    """Volume round trip, then the TTL rule: expired goes, fresh stays."""
    import os

    payload = os.urandom(64_000)
    fresh, stale = new_job_id(), new_job_id()

    put(fresh, "input.mp3", payload)
    put(stale, "input.mp3", b"stale")
    data_vol.commit()

    # Reload so the bytes come back off the Volume rather than out of the
    # container's page cache.
    data_vol.reload()
    checks = {"round_trip_is_byte_exact": get(fresh, "input.mp3") == payload}

    backdate(stale, (DEFAULT_MAX_AGE_HOURS + 1) * 3600)
    removed = cleanup_expired(DEFAULT_MAX_AGE_HOURS)
    checks["expired_job_removed"] = stale in removed
    checks["fresh_job_kept"] = exists(fresh, "input.mp3")

    delete_job(fresh)
    delete_job(stale)
    data_vol.commit()
    return checks


@app.function(image=api_image, timeout=600)
def check_jobs() -> dict:
    """Run one job through the state machine on the real `modal.Dict`."""
    from . import jobs

    job_id = new_job_id()
    checks = {}
    try:
        jobs.create(job_id, "song", params={"semitone_shift": 2})
        checks["record_is_queued"] = jobs.get(job_id)["status"] == jobs.QUEUED

        seen = [
            jobs.update(job_id, status)["progress"]
            for status in (jobs.SEPARATING, jobs.CONVERTING, jobs.MIXING, jobs.DONE)
        ]
        checks["progress_advances_in_order"] = seen == [5, 30, 75, 100]

        try:
            jobs.update(job_id, jobs.CONVERTING)
            checks["done_is_terminal"] = False
        except jobs.JobError:
            checks["done_is_terminal"] = True

        checks["expiry_finds_the_record"] = job_id in jobs.expired_ids(-1)
    finally:
        forgotten = jobs.forget([job_id])
    checks["record_is_removable"] = forgotten == 1 and jobs.find(job_id) is None
    return checks


@app.local_entrypoint()
def main() -> None:
    storage_checks, job_checks = check_storage.remote(), check_jobs.remote()
    checks = {**storage_checks, **job_checks}
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"\n{len(checks)} checks passed")
