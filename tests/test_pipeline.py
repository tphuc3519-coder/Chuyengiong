"""Parameter normalisation and the artifact layout.

The pipeline bodies need Modal (and two GPUs) to run, so what is worth testing
here is `clean_params`, which is where every limit in the plan is applied. It
runs once, before the job record is written, so a value that gets past it is a
value the GPU will be handed and the status endpoint will report.
"""

import pytest

from modal_app import audio_utils as au
from modal_app import jobs, pipeline, storage


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
    job_id = "a" * 32
    jobs.create(job_id, "song", params={"semitone_shift": None}, store=job_store)
    monkeypatch.setattr("modal_app.pitch.suggest_semitone_shift", lambda *a, **k: 9)

    params = {"semitone_shift": None}
    assert pipeline._resolve_shift(job_id, params, b"src", b"ref", "singing") == 9
    # Both the in-flight params handed to the GPU and the record /status reads.
    assert params["semitone_shift"] == 9
    assert jobs.public(job_store[job_id])["semitone_shift"] == 9


def test_a_measured_shift_is_clamped_to_the_mode(monkeypatch, job_store):
    """YIN can suggest a full octave; speech only tolerates ±8."""
    job_id = "a" * 32
    jobs.create(job_id, "speech", params={"semitone_shift": None}, store=job_store)
    monkeypatch.setattr("modal_app.pitch.suggest_semitone_shift", lambda *a, **k: 12)

    params = {"semitone_shift": None}
    assert pipeline._resolve_shift(job_id, params, b"src", b"ref", "speech") == 8


def test_the_song_branch_measures_the_vocal_stem_not_the_mix(monkeypatch, job_store):
    """Measuring the full mix would read the backing track's pitch as well as
    the singer's, which is why this runs after separation."""
    job_id = "a" * 32
    jobs.create(job_id, "song", params={"semitone_shift": None}, store=job_store)
    seen = {}
    monkeypatch.setattr(
        "modal_app.pitch.suggest_semitone_shift",
        lambda source, reference, mode: seen.update(source=source, mode=mode) or 2,
    )

    params = {"semitone_shift": None}
    pipeline._resolve_shift(job_id, params, b"vocal-stem", b"ref", "singing")
    assert seen == {"source": b"vocal-stem", "mode": "singing"}
