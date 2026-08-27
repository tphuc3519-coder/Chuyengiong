"""The web surface, driven through FastAPI's TestClient.

Nothing here touches Modal: `_start_job` is the single seam between the request
and the infrastructure, so patching it lets every validation rule be exercised
against the real routing, form parsing and status codes.

What is worth asserting is the behaviour a client depends on: that a rejected
submit never reaches the pipeline, that the parameters the job is started with
are the clamped ones, and that polling a job that is still running is a clear
409 rather than a 404 or an empty file.
"""

import json
import types

import pytest
from fastapi.testclient import TestClient

from modal_app import api, audit, jobs, pipeline, ratelimit


@pytest.fixture(autouse=True)
def rate_store(monkeypatch):
    """A plain dict behind the rate limit, so no test needs Modal."""
    store: dict = {}
    monkeypatch.setattr(ratelimit, "rate_dict", store)
    return store


@pytest.fixture
def started(monkeypatch):
    """Capture what `/submit` would hand to the pipeline."""
    calls = []

    def fake_start(mode, params, source, reference, client):
        calls.append({"mode": mode, "params": params, "source": source, "reference": reference})
        return "b" * 32

    monkeypatch.setattr(api, "_start_job", fake_start)
    return calls


@pytest.fixture
def client():
    with TestClient(api.web) as test_client:
        yield test_client


def upload(**form):
    """A minimal valid /submit request, overridable field by field."""
    data = {"mode": "song", "consent": "true", **form}
    files = {
        "input": ("song.m4a", b"fake-audio", "audio/mp4"),
        "reference": ("voice.wav", b"fake-voice", "audio/wav"),
    }
    return {"data": data, "files": files}


# --- submit ---------------------------------------------------------------


def test_submit_returns_a_job_id_without_waiting(client, started):
    response = client.post("/submit", **upload())
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "b" * 32,
        "status": jobs.QUEUED,
        "mode": "song",
        # No cap ships, so there is no ceiling to report.
        "jobs_remaining": None,
    }
    assert started[0]["source"] == b"fake-audio"
    assert started[0]["reference"] == b"fake-voice"


def test_submit_without_consent_never_reaches_the_pipeline(client, started):
    """The gate is here, not in the frontend: a checkbox the client enforces on
    its own is not a gate at all."""
    response = client.post("/submit", **upload(consent="false"))
    assert response.status_code == 400
    assert "consent" in response.json()["detail"]
    assert started == []


def test_consent_is_required_even_when_the_field_is_missing(client, started):
    form = upload()
    del form["data"]["consent"]
    assert client.post("/submit", **form).status_code == 400
    assert started == []


def test_parameters_are_clamped_before_the_job_starts(client, started):
    client.post("/submit", **upload(semitone_shift="40", diffusion_steps="900"))
    assert started[0]["params"]["semitone_shift"] == 12
    assert started[0]["params"]["diffusion_steps"] == 100


def test_the_upload_extension_reaches_the_separator(client, started):
    """The artifact is stored as input.mp3; the decoder still needs `.m4a`."""
    client.post("/submit", **upload())
    assert started[0]["params"]["source_ext"] == ".m4a"


def test_an_unknown_mode_is_a_400(client, started):
    response = client.post("/submit", **upload(mode="karaoke"))
    assert response.status_code == 400
    assert started == []


def test_an_unknown_separation_model_is_a_400_not_a_500(client, started):
    """`SeparationError` is a RuntimeError, so it has to be caught by name."""
    response = client.post("/submit", **upload(separation_model="demucs-v9"))
    assert response.status_code == 400
    assert started == []


def test_an_oversized_upload_is_refused_by_size_not_by_content(client, started, monkeypatch):
    monkeypatch.setattr(api, "MAX_INPUT_BYTES", 8)
    response = client.post("/submit", **upload())
    assert response.status_code == 413
    assert started == []


def test_an_empty_upload_is_refused(client, started):
    form = upload()
    form["files"]["input"] = ("song.mp3", b"", "audio/mpeg")
    assert client.post("/submit", **form).status_code == 400
    assert started == []


# --- status ---------------------------------------------------------------


def test_status_reports_progress_without_leaking_params(client, monkeypatch):
    record = jobs.create("c" * 32, "song", params={"semitone_shift": 3}, store={})
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: record)
    body = client.get(f"/status/{'c' * 32}").json()
    assert body["status"] == jobs.QUEUED
    assert body["progress"] == 0
    assert "params" not in body


def test_an_unknown_job_is_a_404(client, monkeypatch):
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: None)
    assert client.get(f"/status/{'d' * 32}").status_code == 404


@pytest.mark.parametrize("job_id", ["..%2f..%2fetc", "x" * 31, "job-1", "e" * 32 + ".."])
def test_a_job_id_that_is_not_a_job_id_never_reaches_storage(client, job_id):
    """`job_id` goes from a URL segment straight to a filesystem path, so it is
    validated before anything opens a file with it."""
    assert client.get(f"/status/{job_id}").status_code == 404


def test_a_malformed_id_and_a_missing_job_do_not_say_the_same_thing(client, monkeypatch):
    """Both are 404s and they used to read identically, which sent someone
    looking for a lost job when they had pasted half an id off a console that
    truncates. A job id is 32 hex characters; saying how many arrived is what
    tells the two apart."""
    truncated = client.get(f"/status/{'d' * 20}")
    assert truncated.status_code == 404
    assert "20" in truncated.json()["detail"], "the reply does not say what was wrong"

    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: None)
    missing = client.get(f"/status/{'d' * 32}")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "no such job"
    assert missing.json()["detail"] != truncated.json()["detail"]


# --- download -------------------------------------------------------------


@pytest.fixture
def volume(monkeypatch):
    """The API container's Volume handle, without Modal behind it."""
    fake = types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(api, "data_vol", fake)
    return fake


def as_status(status, error=None):
    record = jobs.create("e" * 32, "song", store={})
    record["status"] = status
    record["error"] = error
    return record


def test_download_serves_the_mp3_once_the_job_is_done(client, monkeypatch, volume):
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: as_status(jobs.DONE))
    monkeypatch.setattr(api.storage, "get", lambda job_id, name, root=None: b"ID3-mp3")

    response = client.get(f"/download/{'e' * 32}")
    assert response.status_code == 200
    assert response.content == b"ID3-mp3"
    assert response.headers["content-type"] == "audio/mpeg"
    assert "attachment" in response.headers["content-disposition"]


def test_downloading_a_running_job_is_a_409(client, monkeypatch, volume):
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: as_status(jobs.CONVERTING))
    response = client.get(f"/download/{'e' * 32}")
    assert response.status_code == 409
    assert "converting" in response.json()["detail"]


def test_a_failed_job_reports_its_error_rather_than_a_file(client, monkeypatch, volume):
    monkeypatch.setattr(
        jobs, "find", lambda job_id, store=None: as_status(jobs.FAILED, "AudioError: too short")
    )
    response = client.get(f"/download/{'e' * 32}")
    assert response.status_code == 409
    assert response.json()["detail"] == "AudioError: too short"


def test_an_expired_output_is_a_410_not_a_500(client, monkeypatch, volume):
    """The cleanup cron removes files six hours after the last write; the record
    can still be there when the file is not."""

    def gone(job_id, name, root=None):
        raise api.storage.StorageError("no output")

    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: as_status(jobs.DONE))
    monkeypatch.setattr(api.storage, "get", gone)
    assert client.get(f"/download/{'e' * 32}").status_code == 410


def test_an_omitted_shift_reaches_the_pipeline_as_auto(client, started):
    """Plan §7: no slider input means measure it off the vocal stem, which the
    pipeline can only tell from None."""
    assert client.post("/submit", **upload()).status_code == 200
    assert started[0]["params"]["semitone_shift"] is None


def test_an_explicit_zero_is_kept_as_zero(client, started):
    assert client.post("/submit", **upload(semitone_shift="0")).status_code == 200
    assert started[0]["params"]["semitone_shift"] == 0


def test_an_explicit_shift_is_passed_through_clamped(client, started):
    assert client.post("/submit", **upload(mode="speech", semitone_shift="20")).status_code == 200
    assert started[0]["params"]["semitone_shift"] == 8


# --- rate limit -----------------------------------------------------------
#
# The cap ships off, so these turn it on: what they are for is the behaviour a
# deployment gets once `JOBS_PER_HOUR` is set, not what an unset one does.


CAP = ratelimit.PLAN_MAX_JOBS


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setenv(ratelimit.ENV_LIMIT, str(CAP))


def fill_quota(store, address="203.0.113.7"):
    """Use up one client's hourly allowance and return its request headers."""
    key = ratelimit.client_key(address)
    for _ in range(CAP):
        ratelimit.check(key, store=store, limit=CAP)
    return {"x-forwarded-for": address}


def test_a_client_over_its_hourly_cap_is_a_429(client, started, rate_store, capped):
    """Plan §9: 5 jobs an hour. The cap is here because the browser uploads
    straight to Modal, so the frontend is not on this path at all."""
    headers = fill_quota(rate_store)
    response = client.post("/submit", **upload(), headers=headers)
    assert response.status_code == 429
    assert started == []


def test_retry_after_is_the_real_wait_not_a_whole_window(client, started, rate_store, capped):
    """The oldest request is what frees the next slot, so a client ten minutes
    in should be told fifty, not sixty."""
    headers = fill_quota(rate_store)
    key = ratelimit.client_key(headers["x-forwarded-for"])
    rate_store[key] = [stamp - 600 for stamp in rate_store[key]]

    response = client.post("/submit", **upload(), headers=headers)
    assert response.status_code == 429
    wait = int(response.headers["retry-after"])
    assert 0 < wait <= ratelimit.WINDOW_SEC - 599


def test_the_cap_is_per_client_not_global(client, started, rate_store, capped):
    fill_quota(rate_store, "203.0.113.7")
    response = client.post("/submit", **upload(), headers={"x-forwarded-for": "198.51.100.9"})
    assert response.status_code == 200
    assert len(started) == 1


def test_a_rejected_submit_does_not_spend_a_slot(client, started, rate_store, capped):
    """Validation runs after the quota is read but before it is spent, so a
    client that sends a bad request can fix it and retry."""
    address = "203.0.113.7"
    headers = {"x-forwarded-for": address}
    assert client.post("/submit", **upload(consent="false"), headers=headers).status_code == 400
    assert ratelimit.remaining(ratelimit.client_key(address), store=rate_store) == (CAP)


def test_the_slot_is_spent_when_the_job_actually_starts(client, monkeypatch, rate_store, capped):
    """`_start_job` is the real thing here, with only the infrastructure stubbed."""
    address = "203.0.113.7"
    monkeypatch.setattr(api.storage, "put", lambda *a, **k: "")
    monkeypatch.setattr(api.data_vol, "commit", lambda: None)
    monkeypatch.setattr(jobs, "create", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "spawn", lambda *a, **k: None)

    response = client.post("/submit", **upload(), headers={"x-forwarded-for": address})
    assert response.status_code == 200
    assert response.json()["jobs_remaining"] == CAP - 1
    assert ratelimit.remaining(ratelimit.client_key(address), store=rate_store) == (CAP - 1)


# --- wiring ---------------------------------------------------------------


def test_the_api_still_starts_without_the_audio_stack():
    """`api.py` may import the pipeline, but not torch: the web container has
    no GPU image and importing conversion at module scope would need one."""
    assert not hasattr(api, "VoiceConverter")
    assert pipeline.run_song_pipeline.info.function_name == "run_song_pipeline"


# --- audit trail ----------------------------------------------------------


def audit_lines(captured: str) -> list[dict]:
    return [
        json.loads(line[len(audit.PREFIX) + 1 :])
        for line in captured.splitlines()
        if line.startswith(audit.PREFIX)
    ]


def test_a_submit_is_audited_without_naming_the_files(client, started, capsys):
    """Plan §8 item 5: job id, timestamp, and that the gate was passed — the
    trail a complaint would be answered from. Nothing about the audio."""
    client.post("/submit", **upload(), headers={"x-forwarded-for": "198.51.100.4"})
    entry = audit_lines(capsys.readouterr().out)[-1]

    assert entry["event"] == audit.SUBMIT
    assert entry["job"] == "b" * 32
    assert entry["ts"].endswith("Z")
    assert entry["consent"] is True
    assert (entry["input_bytes"], entry["reference_bytes"]) == (10, 10)
    # The address is hashed by `ratelimit`, and neither it nor the file names
    # the client sent appear anywhere in the line.
    assert entry["client"] == ratelimit.client_key("198.51.100.4")
    printed = json.dumps(entry)
    assert "198.51.100.4" not in printed
    assert "song.m4a" not in printed and "voice.wav" not in printed


def test_a_refused_submit_leaves_no_audit_line(client, started, capsys):
    """No job started, nothing to trace: the trail is of jobs, not of requests."""
    client.post("/submit", **upload(consent="false"))
    assert audit_lines(capsys.readouterr().out) == []


def test_a_download_is_audited(client, monkeypatch, volume, capsys):
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: as_status(jobs.DONE))
    monkeypatch.setattr(api.storage, "get", lambda job_id, name, root=None: b"ID3-mp3")

    client.get(f"/download/{'e' * 32}")
    entry = audit_lines(capsys.readouterr().out)[-1]
    assert (entry["event"], entry["job"], entry["output_bytes"]) == (
        audit.DOWNLOAD,
        "e" * 32,
        7,
    )


def test_a_download_that_serves_nothing_is_not_audited(client, monkeypatch, volume, capsys):
    monkeypatch.setattr(jobs, "find", lambda job_id, store=None: as_status(jobs.CONVERTING))
    client.get(f"/download/{'e' * 32}")
    assert audit_lines(capsys.readouterr().out) == []
