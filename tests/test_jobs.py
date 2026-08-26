"""Job state machine, driven with a plain dict in place of the modal.Dict."""

import time

import pytest

from modal_app import audio_utils as au
from modal_app import jobs


@pytest.fixture
def store():
    return {}


def new(store, mode="song", **kw):
    job_id = "0" * 31 + "1"
    return jobs.create(job_id, mode, store=store, **kw)


def test_create_starts_queued_at_zero(store):
    record = new(store, params={"semitone_shift": 3})

    assert record["status"] == jobs.QUEUED
    assert record["progress"] == 0
    assert record["error"] is None
    assert record["params"] == {"semitone_shift": 3}
    assert record["created_at"] == record["updated_at"]
    assert jobs.get(record["id"], store=store) == record


def test_create_rejects_a_duplicate_id(store):
    new(store)
    with pytest.raises(jobs.JobError):
        new(store)


def test_create_rejects_an_unknown_mode(store):
    with pytest.raises(jobs.JobError):
        new(store, mode="singing")  # the conversion mode, not a job mode


def test_song_runs_the_full_machine(store):
    job_id = new(store, mode="song")["id"]
    seen = []
    for status in (jobs.SEPARATING, jobs.CONVERTING, jobs.MIXING, jobs.DONE):
        seen.append(jobs.update(job_id, status, store=store)["progress"])

    assert seen == [5, 30, 75, 100]
    assert jobs.get(job_id, store=store)["status"] == jobs.DONE


def test_speech_skips_separation_and_mixing(store):
    """`speech` has no stems to separate and nothing to mix back in."""
    job_id = new(store, mode="speech")["id"]
    jobs.update(job_id, jobs.CONVERTING, store=store)
    record = jobs.update(job_id, jobs.DONE, store=store)
    assert record["status"] == jobs.DONE
    assert record["progress"] == 100


def test_transitions_cannot_go_backwards(store):
    job_id = new(store)["id"]
    jobs.update(job_id, jobs.CONVERTING, store=store)
    for status in (jobs.SEPARATING, jobs.CONVERTING, jobs.QUEUED):
        with pytest.raises(jobs.JobError):
            jobs.update(job_id, status, store=store)


def test_terminal_states_are_terminal(store):
    job_id = new(store)["id"]
    jobs.update(job_id, jobs.CONVERTING, store=store)
    jobs.update(job_id, jobs.DONE, store=store)
    with pytest.raises(jobs.JobError):
        jobs.update(job_id, jobs.MIXING, store=store)


def test_unknown_status_is_rejected(store):
    job_id = new(store)["id"]
    with pytest.raises(jobs.JobError):
        jobs.update(job_id, "uploading", store=store)


def test_unknown_job_is_reported_not_invented(store):
    assert jobs.find("f" * 32, store=store) is None
    with pytest.raises(jobs.JobError):
        jobs.get("f" * 32, store=store)


def test_failure_is_reachable_from_any_live_state(store):
    for status in (jobs.QUEUED, jobs.SEPARATING, jobs.CONVERTING, jobs.MIXING):
        local = {}
        job_id = new(local)["id"]
        if status != jobs.QUEUED:
            jobs.update(job_id, status, store=local)
        record = jobs.fail(job_id, "gpu fell over", store=local)
        assert record["status"] == jobs.FAILED
        assert record["error"] == "gpu fell over"


def test_failing_twice_does_not_raise(store):
    """The pipeline's `except` block must not explode while reporting."""
    job_id = new(store)["id"]
    jobs.fail(job_id, "first", store=store)
    assert jobs.fail(job_id, "second", store=store)["error"] == "first"


def test_failure_always_carries_a_message(store):
    job_id = new(store)["id"]
    record = jobs.update(job_id, jobs.FAILED, store=store)
    assert record["error"]


def test_progress_never_decreases(store):
    job_id = new(store)["id"]
    jobs.update(job_id, jobs.CONVERTING, store=store)
    assert jobs.update(job_id, progress=60, store=store)["progress"] == 60
    assert jobs.update(job_id, progress=10, store=store)["progress"] == 60
    assert jobs.update(job_id, progress=1000, store=store)["progress"] == 100


def test_progress_within_a_state_moves_without_a_transition(store):
    job_id = new(store)["id"]
    jobs.update(job_id, jobs.CONVERTING, store=store)
    record = jobs.update(job_id, progress=55, store=store)
    assert record["status"] == jobs.CONVERTING
    assert record["progress"] == 55


def test_updated_at_moves_with_every_write(store):
    job_id = new(store, now=1000.0)["id"]
    record = jobs.update(job_id, jobs.SEPARATING, store=store, now=1042.0)
    assert record["created_at"] == 1000.0
    assert record["updated_at"] == 1042.0


def test_public_view_hides_params(store):
    record = new(store, params={"reference_path": "/data/x/reference.wav"})
    view = jobs.public(record)
    assert "params" not in view
    assert set(view) == {
        "id",
        "status",
        "progress",
        "mode",
        "error",
        "created_at",
        "updated_at",
    }


def test_expired_records_are_found_and_forgotten(store):
    now = time.time()
    old = jobs.create("a" * 32, "song", store=store, now=now - 7 * 3600)["id"]
    fresh = jobs.create("b" * 32, "song", store=store, now=now)["id"]

    assert jobs.expired_ids(6, store=store) == [old]
    assert jobs.forget([old, "c" * 32], store=store) == 1
    assert jobs.find(old, store=store) is None
    assert jobs.find(fresh, store=store) is not None


def test_job_modes_map_onto_conversion_modes():
    """`song` is a job mode; `singing` is the checkpoint it converts with."""
    assert set(jobs.CONVERSION_MODE) == set(jobs.JOB_MODES)
    assert set(jobs.CONVERSION_MODE.values()) <= set(au.MODES)
    assert jobs.CONVERSION_MODE["song"] == "singing"


def test_progress_is_monotonic_along_the_machine():
    assert [jobs.PROGRESS[s] for s in jobs.ORDER] == sorted(jobs.PROGRESS[s] for s in jobs.ORDER)
    assert jobs.PROGRESS[jobs.DONE] == 100
