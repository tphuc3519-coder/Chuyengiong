"""The web surface, driven through FastAPI's TestClient.

Nothing here touches Modal: `_start_job` is the single seam between the request
and the infrastructure, so patching it lets every validation rule be exercised
against the real routing, form parsing and status codes.

What is worth asserting is the behaviour a client depends on: that a rejected
submit never reaches the pipeline, that the parameters the job is started with
are the clamped ones, and that polling a job that is still running is a clear
409 rather than a 404 or an empty file.
"""

import types

import pytest
from fastapi.testclient import TestClient

from modal_app import api, jobs, pipeline


@pytest.fixture
def started(monkeypatch):
    """Capture what `/submit` would hand to the pipeline."""
    calls = []

    def fake_start(mode, params, source, reference):
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
    assert response.json() == {"job_id": "b" * 32, "status": jobs.QUEUED, "mode": "song"}
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


# --- wiring ---------------------------------------------------------------


def test_the_api_still_starts_without_the_audio_stack():
    """`api.py` may import the pipeline, but not torch: the web container has
    no GPU image and importing conversion at module scope would need one."""
    assert not hasattr(api, "VoiceConverter")
    assert pipeline.run_song_pipeline.info.function_name == "run_song_pipeline"
