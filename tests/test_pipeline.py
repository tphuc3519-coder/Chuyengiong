"""Parameter normalisation and the artifact layout.

The pipeline bodies need Modal (and two GPUs) to run, so what is worth testing
here is `clean_params`, which is where every limit in the plan is applied. It
runs once, before the job record is written, so a value that gets past it is a
value the GPU will be handed and the status endpoint will report.
"""

import json

import pytest

from modal_app import audio_utils as au
from modal_app import audit, jobs, pipeline, storage, watermark


def test_defaults_come_from_the_mode():
    song = pipeline.clean_params("song")
    speech = pipeline.clean_params("speech")
    assert song["diffusion_steps"] == au.DEFAULT_DIFFUSION_STEPS["singing"] == 50
    assert speech["diffusion_steps"] == au.DEFAULT_DIFFUSION_STEPS["speech"] == 25
    assert song["vocal_gain_db"] == 0.0
    # None, not 0: nothing was asked for, so the pipeline measures it (§7).
    assert song["semitone_shift"] is None
    assert speech["semitone_shift"] is None


def test_an_explicit_zero_is_not_auto_detect():
    """The distinction the whole feature rests on: 0 means "leave the pitch
    alone", None means "work it out from the audio"."""
    assert pipeline.clean_params("song", {"semitone_shift": 0})["semitone_shift"] == 0
    assert pipeline.clean_params("song", {"semitone_shift": None})["semitone_shift"] is None


def test_speech_gets_the_narrower_pitch_range():
    """±8 for speech: a bigger shift smears tone in tonal languages. Singing
    carries the melody, so it keeps ±12."""
    assert pipeline.clean_params("song", {"semitone_shift": 20})["semitone_shift"] == 12
    assert pipeline.clean_params("speech", {"semitone_shift": 20})["semitone_shift"] == 8
    assert pipeline.clean_params("speech", {"semitone_shift": -20})["semitone_shift"] == -8


def test_diffusion_steps_are_clamped_not_rejected():
    assert pipeline.clean_params("song", {"diffusion_steps": 5})["diffusion_steps"] == 10
    assert pipeline.clean_params("song", {"diffusion_steps": 500})["diffusion_steps"] == 100
    assert pipeline.clean_params("song", {"diffusion_steps": 30})["diffusion_steps"] == 30


def test_vocal_gain_is_clamped():
    assert pipeline.clean_params("song", {"vocal_gain_db": 99})["vocal_gain_db"] == 12.0


def test_only_the_song_branch_carries_a_separation_model():
    assert pipeline.clean_params("song")["separation_model"] == "roformer"
    assert "separation_model" not in pipeline.clean_params("speech")


def test_an_unknown_separation_model_is_refused():
    from modal_app.separation import SeparationError

    with pytest.raises(SeparationError):
        pipeline.clean_params("song", {"separation_model": "demucs-v9"})


def test_an_unknown_mode_is_refused():
    with pytest.raises(jobs.JobError):
        pipeline.clean_params("singing")  # the conversion mode, not a job mode


def test_the_source_extension_is_carried_through_for_the_decoder():
    """audio-separator picks its decoder by file name, so the real extension
    has to survive the trip even though the artifact is called input.mp3."""
    assert pipeline.clean_params("song", {"source_ext": ".m4a"})["source_ext"] == ".m4a"
    assert pipeline.clean_params("song", {"source_ext": ".exe"})["source_ext"] == ".mp3"


def test_params_survive_a_round_trip_through_a_job_record():
    """They are stored in `modal.Dict` and read back in another container."""
    store = {}
    params = pipeline.clean_params("song", {"semitone_shift": 3})
    jobs.create("a" * 32, "song", params=params, store=store)
    assert jobs.get("a" * 32, store)["params"] == params


def test_every_artifact_name_is_one_storage_accepts():
    for name in (
        pipeline.INPUT,
        pipeline.REFERENCE,
        pipeline.VOCAL,
        pipeline.INSTRUMENTAL,
        pipeline.CONVERTED,
        pipeline.OUTPUT,
    ):
        assert storage.check_name(name) == name


def test_failure_text_is_short_enough_to_render():
    text = pipeline._error_text(RuntimeError("x" * 5000))
    assert text.startswith("RuntimeError: ")
    assert len(text) <= pipeline.MAX_ERROR_CHARS


# --- the pitch shift (plan §7) --------------------------------------------


@pytest.fixture
def job_store(monkeypatch):
    """A job record on a plain dict, with `record_params` writing into it.

    `_resolve_shift` calls `jobs.record_params` with the default store, which
    is the live `modal.Dict`; binding the original to a local dict is what
    keeps these tests off Modal.
    """
    store: dict = {}
    original = jobs.record_params
    monkeypatch.setattr(
        jobs, "record_params", lambda jid, updates: original(jid, updates, store=store)
    )
    return store


def test_an_explicit_shift_is_used_without_measuring_anything(monkeypatch):
    """Auto-detect decodes two files and runs YIN over both. A client that
    already said what it wants should not pay for that."""
    called = []
    monkeypatch.setattr(
        "modal_app.pitch.suggest_semitone_shift",
        lambda *a, **k: called.append(a) or 3,
    )
    params = {"semitone_shift": -4}
    assert pipeline._resolve_shift("a" * 32, params, b"src", b"ref", "singing") == -4
    assert called == []


def test_an_absent_shift_is_measured_and_written_back(monkeypatch, job_store):
    """Speech, because speech is what auto-detect is for — see `AUTO_DETECT_MODES`."""
    job_id = "a" * 32
    jobs.create(job_id, "speech", params={"semitone_shift": None}, store=job_store)
    monkeypatch.setattr("modal_app.pitch.suggest_semitone_shift", lambda *a, **k: 6)

    params = {"semitone_shift": None}
    assert pipeline._resolve_shift(job_id, params, b"src", b"ref", "speech") == 6
    # Both the in-flight params handed to the GPU and the record /status reads.
    assert params["semitone_shift"] == 6
    assert jobs.public(job_store[job_id])["semitone_shift"] == 6


def test_a_measured_shift_is_clamped_to_the_mode(monkeypatch, job_store):
    """YIN can suggest a full octave; speech only tolerates ±8."""
    job_id = "a" * 32
    jobs.create(job_id, "speech", params={"semitone_shift": None}, store=job_store)
    monkeypatch.setattr("modal_app.pitch.suggest_semitone_shift", lambda *a, **k: 12)

    params = {"semitone_shift": None}
    assert pipeline._resolve_shift(job_id, params, b"src", b"ref", "speech") == 8


def test_a_song_keeps_the_key_it_was_written_in(monkeypatch, job_store):
    """The bug this exists for. The converted vocal is mixed back over an
    instrumental nothing here transposes, so a shift that is not a whole octave
    puts the singer in a different key from the backing track. Measuring one
    made it certain: the distance from a singer's median F0 to a *speaking*
    reference is a musically arbitrary number."""
    job_id = "a" * 32
    jobs.create(job_id, "song", params={"semitone_shift": None}, store=job_store)
    called = []
    monkeypatch.setattr(
        "modal_app.pitch.suggest_semitone_shift",
        lambda *a, **k: called.append(a) or 7,
    )

    params = {"semitone_shift": None}
    assert pipeline._resolve_shift(job_id, params, b"vocal-stem", b"ref", "singing") == 0
    assert called == [], "a song measured a shift it should not have"
    # /status still reports what was applied, and 0 is what was applied.
    assert jobs.public(job_store[job_id])["semitone_shift"] == 0


def test_a_song_still_takes_a_shift_the_user_asked_for(monkeypatch, job_store):
    """Off by default is not the same as unavailable: the slider still moves,
    and an octave is the one that stays in key."""
    monkeypatch.setattr("modal_app.pitch.suggest_semitone_shift", lambda *a, **k: 7)
    params = {"semitone_shift": -12}
    assert pipeline._resolve_shift("a" * 32, params, b"vocal", b"ref", "singing") == -12


# --- the audit trail (plan §8 item 5) -------------------------------------


def last_audit(captured: str) -> dict:
    line = [ln for ln in captured.splitlines() if ln.startswith(audit.PREFIX)][-1]
    return json.loads(line[len(audit.PREFIX) + 1 :])


def test_a_finished_run_records_what_it_did(capsys):
    job_id = "a" * 32
    params = {"semitone_shift": 5, "diffusion_steps": 50, "separation_model": "htdemucs"}
    pipeline._finished(job_id, "song", params, started=0.0)

    entry = last_audit(capsys.readouterr().out)
    assert entry["event"] == audit.DONE
    assert (entry["job"], entry["mode"], entry["shift"], entry["steps"]) == (
        job_id,
        "song",
        5,
        50,
    )
    assert entry["seconds"] > 0
    assert entry["reason"] is None


def test_a_failure_records_the_exception_class_not_its_message(capsys):
    """The message can quote ffmpeg on the user's own file; `/status` carries
    it to the one person entitled to read it, and the log does not."""
    exc = RuntimeError("cannot decode /data/deadbeef/input.mp3: bí mật")
    pipeline._finished("a" * 32, "speech", {"semitone_shift": None}, started=0.0, exc=exc)

    entry = last_audit(capsys.readouterr().out)
    assert entry["event"] == audit.FAILED
    assert entry["reason"] == "RuntimeError"
    assert "bí mật" not in json.dumps(entry, ensure_ascii=False)


# --- watermarking (plan §8, "Cân nhắc thêm") -------------------------------


def test_a_job_records_whether_it_was_watermarked(monkeypatch):
    """Resolved when the job is created, not when the mix runs: the record has
    to say what was done to this file, not what the config says a week later."""
    monkeypatch.delenv(watermark.ENV_FLAG, raising=False)
    assert pipeline.clean_params("song")["watermark"] is True
    monkeypatch.setenv(watermark.ENV_FLAG, "0")
    assert pipeline.clean_params("song")["watermark"] is False
    assert pipeline.clean_params("speech")["watermark"] is False


def test_watermarking_is_not_a_client_setting(monkeypatch):
    """It is deployment config. A caller asking for no watermark is ignored."""
    monkeypatch.delenv(watermark.ENV_FLAG, raising=False)
    assert pipeline.clean_params("song", {"watermark": False})["watermark"] is True


def test_no_watermark_hook_is_built_when_it_is_switched_off():
    """`mixing` takes None to mean "encode and ship", so this is the whole
    off-switch: no import of the model module, no container started."""
    assert pipeline._watermark("j" * 32, {"watermark": False}) is None


def test_the_audit_line_says_whether_the_output_was_watermarked():
    """A complaint arrives with a file; the log has to say whether that file
    should carry a watermark at all."""
    line = audit.event_line(audit.DONE, "b" * 32, mode="song", watermark=True)
    assert json.loads(line.split(" ", 1)[1])["watermark"] is True


# --- the browser's patience vs the server's ------------------------------


def test_the_browser_does_not_give_up_before_the_pipeline_does():
    """These two drifted apart once and cost a real job.

    A client that stops polling first tells the user a *running* job failed and
    to try again — and trying again starts a second GPU job beside the first,
    doubling the bill and spending another slot off the hourly cap. Only the
    server may decide a job is over, so its timeout is the floor for the
    browser's. A comment was all that held them together last time, which is
    why this is a test.
    """
    import re
    from pathlib import Path

    api_ts = Path(__file__).resolve().parent.parent / "web" / "lib" / "api.ts"
    source = api_ts.read_text()

    match = re.search(r"POLL_TIMEOUT_MS\s*=\s*(\d+)\s*\*\s*60_000", source)
    assert match, "POLL_TIMEOUT_MS is not declared as `<minutes> * 60_000` any more"

    browser_sec = int(match.group(1)) * 60
    assert browser_sec >= pipeline.PIPELINE_TIMEOUT, (
        f"the browser gives up at {browser_sec}s but the pipeline runs to "
        f"{pipeline.PIPELINE_TIMEOUT}s"
    )
