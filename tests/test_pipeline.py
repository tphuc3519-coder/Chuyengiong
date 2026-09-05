"""Parameter normalisation and the artifact layout.

The pipeline bodies need Modal (and two GPUs) to run, so what is worth testing
here is `clean_params`, which is where every limit in the plan is applied. It
runs once, before the job record is written, so a value that gets past it is a
value the GPU will be handed and the status endpoint will report.
"""

import json
import types

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


def test_the_new_knobs_default_to_the_middle_and_clamp_at_the_ends():
    """Both are `None`-means-default rather than falsy-means-default: 0 is a
    real setting for each of them — no guidance, and no post-processing — and
    `or` would have swallowed it."""
    from modal_app import enhance

    plain = pipeline.clean_params("speech")
    assert plain["cfg_rate"] == au.DEFAULT_CFG_RATE
    assert plain["clarity"] == enhance.DEFAULT_CLARITY
    assert pipeline.clean_params("speech", {"cfg_rate": 0, "clarity": 0})["cfg_rate"] == 0.0
    assert pipeline.clean_params("speech", {"cfg_rate": 0, "clarity": 0})["clarity"] == 0.0
    assert pipeline.clean_params("speech", {"cfg_rate": 9})["cfg_rate"] == au.CFG_RATE_MAX
    assert pipeline.clean_params("speech", {"clarity": -3})["clarity"] == enhance.CLARITY_MIN


def test_an_unusable_voice_name_becomes_no_voice_rather_than_an_error():
    """Zero shot is what every job did before profiles existed, so a name that
    could not be one is not worth failing over. A name that *is* usable and has
    no profile behind it fails in the GPU container, which is the only place
    that can be known."""
    assert pipeline.clean_params("speech")["voice_profile"] == ""
    assert pipeline.clean_params("speech", {"voice_profile": "../etc"})["voice_profile"] == ""
    assert pipeline.clean_params("speech", {"voice_profile": "mai"})["voice_profile"] == "mai"


def test_the_beat_branch_separates_like_a_song_and_carries_its_own_params():
    """It is `song` with the instrumental replaced, so it runs the separator and
    converts as singing — and the key is left alone, because the bed is being
    fitted to the singer rather than the singer to the bed."""
    params = pipeline.clean_params("beat", {"beat_prompt": "  boom bap, 90 BPM  "})
    assert jobs.CONVERSION_MODE["beat"] == "singing"
    assert params["separation_model"] == "roformer"
    assert params["beat_prompt"] == "boom bap, 90 BPM"
    assert params["beat_seed"] == -1
    assert pipeline.clean_params("beat", {"semitone_shift": 20})["semitone_shift"] == 12


def test_an_absent_beat_prompt_means_a_beat_was_uploaded():
    """Empty is a real state on this branch and not a missing value. `api.submit`
    is where each source's requirements are checked, because it is the only
    layer that can see an upload."""
    assert pipeline.clean_params("beat")["beat_prompt"] == ""


def test_the_beat_source_is_named_rather_than_inferred():
    """Three genuinely different things, and two of them send no file — so
    there is nothing to infer the source from. `derive` sends no description
    either: it reads one off the song."""
    assert pipeline.clean_params("beat")["beat_source"] == pipeline.DEFAULT_BEAT_SOURCE
    for source in pipeline.BEAT_SOURCES:
        assert pipeline.clean_params("beat", {"beat_source": source})["beat_source"] == source


def test_an_unknown_beat_source_falls_back_to_the_one_needing_no_model():
    """Upload needs no GPU and no weights, so it is the safe answer — and
    `api.submit` refuses it anyway when no file came with it."""
    assert (
        pipeline.clean_params("beat", {"beat_source": "telepathy"})["beat_source"]
        == pipeline.DEFAULT_BEAT_SOURCE
    )


def test_every_beat_source_is_one_of_the_three_that_reach_the_bar():
    """`remake` — the song's own chords played back on synthesised instruments,
    and nothing after them — is gone. It could not reach the quality it was
    being held against.

    `derive` is not that feature returning. The same chart now goes to Stable
    Audio Open as `init_audio` and comes back played on real instruments, which
    is the difference between synthesis being the output and synthesis being
    the instruction."""
    assert set(pipeline.BEAT_SOURCES) == {"upload", "generate", "derive"}


def test_only_the_sources_with_no_file_of_their_own_cost_a_gpu():
    assert set(pipeline.GENERATING_SOURCES) == {"generate", "derive"}
    assert "upload" not in pipeline.GENERATING_SOURCES


def test_an_unrecognised_init_clamps_to_the_branch_that_copies_nothing():
    """Which way this clamps is the whole point.

    `original` shows the model the song's own master; `sketch` shows it this
    repository's oscillators. A typo, an old client or a hand-written form
    should land on the one that cannot produce a derivative work."""
    assert pipeline.DEFAULT_BEAT_INIT == "sketch"
    for value in ("", "ORIGINAL_", "nonsense", None, "  "):
        assert pipeline.clean_params("beat", {"beat_init": value})["beat_init"] == "sketch"
    assert pipeline.clean_params("beat", {"beat_init": "original"})["beat_init"] == "original"
    assert pipeline.clean_params("beat", {"beat_init": " Original "})["beat_init"] == "original"


def test_a_beat_prompt_cannot_be_longer_than_the_generator_accepts():
    params = pipeline.clean_params("beat", {"beat_prompt": "x" * 5000})
    assert len(params["beat_prompt"]) == pipeline.BEAT_PROMPT_CHARS


def test_a_nonsense_seed_falls_back_to_a_different_beat_every_time():
    assert pipeline.clean_params("beat", {"beat_seed": "lucky"})["beat_seed"] == -1
    assert pipeline.clean_params("beat", {"beat_seed": 7})["beat_seed"] == 7


def test_only_the_branches_that_start_from_a_mix_run_the_separator():
    assert set(pipeline.SEPARATING_MODES) == {"song", "beat", "rebeat"}
    for mode in pipeline.SEPARATING_MODES:
        assert "separation_model" in pipeline.clean_params(mode)


def test_both_beat_branches_take_a_beat_source():
    assert set(pipeline.BEAT_MODES) == {"beat", "rebeat"}
    for mode in pipeline.BEAT_MODES:
        assert pipeline.clean_params(mode)["beat_source"] == pipeline.DEFAULT_BEAT_SOURCE


def test_the_rebeat_branch_converts_nothing_and_says_so_by_omission():
    """It keeps the singer it was given, so a pitch shift, a step count and a
    voice profile are not settings it has. Recording them anyway would put
    numbers in the job record that nothing reads and `/status` reports."""
    params = pipeline.clean_params("rebeat")
    assert "rebeat" not in jobs.CONVERSION_MODE
    for absent in ("semitone_shift", "diffusion_steps", "cfg_rate", "voice_profile"):
        assert absent not in params
    # What it does still have: it separates, it mixes, and it is watermarked.
    for present in ("separation_model", "beat_source", "clarity", "vocal_gain_db", "watermark"):
        assert present in params


def test_the_vocal_branch_converts_as_singing_and_keeps_the_key():
    """It is the `song` branch without the separator, so it gets the singing
    checkpoint's wider pitch range — and, like a song, it is not measured
    against the reference: a take has a key and the point is to keep it."""
    assert jobs.CONVERSION_MODE["vocal"] == "singing"
    assert pipeline.clean_params("vocal", {"semitone_shift": 20})["semitone_shift"] == 12
    assert (
        pipeline.clean_params("vocal")["diffusion_steps"] == au.DEFAULT_DIFFUSION_STEPS["singing"]
    )
    assert "singing" not in pipeline.AUTO_DETECT_MODES


def test_only_the_song_branch_carries_a_separation_model():
    assert pipeline.clean_params("song")["separation_model"] == "roformer"
    assert "separation_model" not in pipeline.clean_params("speech")
    # `vocal` is the branch that exists to not run it.
    assert "separation_model" not in pipeline.clean_params("vocal")


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


# --- resuming after a restart ---------------------------------------------


@pytest.fixture
def volume_root(tmp_path, monkeypatch):
    """Artifacts on a real directory, with the Volume handle stubbed out."""
    monkeypatch.setattr(storage, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline, "data_vol", types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    return tmp_path


def test_an_artifact_that_is_there_is_reused(volume_root):
    job_id = "a" * 32
    assert pipeline._done_already(job_id, pipeline.VOCAL) is None
    storage.put(job_id, pipeline.VOCAL, b"stem-bytes")
    assert pipeline._done_already(job_id, pipeline.VOCAL) == b"stem-bytes"


def test_a_half_written_artifact_is_not_mistaken_for_a_finished_one(volume_root):
    """The property the whole resume rests on. `storage.put` writes to a temp
    name and renames, so a stage killed mid-write leaves a `.part` and nothing
    a later run can pick up by mistake."""
    job_id = "a" * 32
    (volume_root / job_id).mkdir()
    (volume_root / job_id / f"{pipeline.VOCAL}.part").write_bytes(b"half a stem")
    assert pipeline._done_already(job_id, pipeline.VOCAL) is None


def test_a_restarted_song_does_not_pay_for_the_gpu_twice(volume_root, monkeypatch, job_store):
    """What this is all for. A preemption while converting used to restart the
    pipeline at the top and separate the whole song again, on a GPU, for work
    already sitting on the Volume."""
    job_id = "a" * 32
    jobs.create(job_id, "song", params={"semitone_shift": 0}, store=job_store)
    monkeypatch.setattr(jobs, "job_dict", job_store)
    monkeypatch.setattr(pipeline, "_finished", lambda *a, **k: None)

    for name, payload in (
        (pipeline.INPUT, b"song"),
        (pipeline.REFERENCE, b"voice"),
        (pipeline.VOCAL, b"vocal-stem"),
        (pipeline.INSTRUMENTAL, b"instrumental-stem"),
        (pipeline.CONVERTED, b"converted-vocal"),
    ):
        storage.put(job_id, name, payload)

    used = {}

    def no_gpu(*a, **k):
        raise AssertionError("a restart paid for the GPU again")

    monkeypatch.setattr("modal_app.separation.Separator", no_gpu)
    monkeypatch.setattr("modal_app.conversion.VoiceConverter", no_gpu)
    monkeypatch.setattr(
        "modal_app.mixing.mix",
        lambda vocal, instrumental, **k: (
            used.update(vocal=vocal, instrumental=instrumental) or b"final-mp3"
        ),
    )

    pipeline.run_song_pipeline.local(job_id, pipeline.clean_params("song"))

    # The mix ran on what was already there, and only the mix.
    assert used == {"vocal": b"converted-vocal", "instrumental": b"instrumental-stem"}
    assert storage.get(job_id, pipeline.OUTPUT) == b"final-mp3"
    assert jobs.get(job_id, job_store)["status"] == jobs.DONE


def test_a_restart_redoes_only_what_is_missing(volume_root, monkeypatch, job_store):
    """The common case: preempted during conversion, so the stems survived and
    the converted vocal did not. Separation is skipped, conversion is not."""
    job_id = "a" * 32
    jobs.create(job_id, "song", params={"semitone_shift": 0}, store=job_store)
    monkeypatch.setattr(jobs, "job_dict", job_store)
    monkeypatch.setattr(pipeline, "_finished", lambda *a, **k: None)

    for name, payload in (
        (pipeline.INPUT, b"song"),
        (pipeline.REFERENCE, b"voice"),
        (pipeline.VOCAL, b"vocal-stem"),
        (pipeline.INSTRUMENTAL, b"instrumental-stem"),
    ):
        storage.put(job_id, name, payload)

    ran = {"separate": False, "convert": False}

    def separator(**_):
        raise AssertionError("stems were on the Volume and it separated anyway")

    def converter(**_):
        return types.SimpleNamespace(
            convert=types.SimpleNamespace(
                remote=lambda **kw: ran.update(convert=True, source=kw["source_wav"]) or b"fresh"
            )
        )

    monkeypatch.setattr("modal_app.separation.Separator", separator)
    monkeypatch.setattr("modal_app.conversion.VoiceConverter", converter)
    monkeypatch.setattr("modal_app.mixing.mix", lambda vocal, instrumental, **k: b"final-mp3")

    pipeline.run_song_pipeline.local(job_id, pipeline.clean_params("song"))

    assert ran["convert"], "the missing stage did not run"
    # And it converted the stem that was already there, not a fresh one.
    assert ran["source"] == b"vocal-stem"
    assert storage.get(job_id, pipeline.CONVERTED) == b"fresh"
    assert jobs.get(job_id, job_store)["status"] == jobs.DONE


# --- the class the pipeline actually calls --------------------------------


def test_the_generator_is_reached_through_the_lookup_not_the_imported_class(
    volume_root, monkeypatch, job_store
):
    """The bug this pins produced, three minutes into a paid job:

        AttributeError: 'function' object has no attribute 'remote'

    `app.cls(...)` returns a *new* object and leaves `BeatGenerator` alone, so
    the name imported from `beatgen` is a plain class and `.remote` on its
    bound method never existed. `beatgen.generator()` is the accessor that
    knows this; `_generate_beat` has to go through it.
    """
    from modal_app import beatgen

    job_id = "b" * 32
    jobs.create(job_id, "beat", params={}, store=job_store)
    monkeypatch.setattr(jobs, "job_dict", job_store)

    called = {}

    class Fake:
        def __call__(self):
            return self

        @property
        def generate(self):
            return self

        def remote(self, **kwargs):
            called.update(kwargs)
            return b"generated-beat"

    monkeypatch.setattr(beatgen, "generator", lambda: Fake())
    params = pipeline.clean_params("beat", {"beat_source": "generate", "beat_prompt": "boom bap"})

    assert pipeline._generate_beat(job_id, params) == b"generated-beat"
    assert called["prompt"] == "boom bap"
    # `generate` and `derive` differ only in these two.
    assert called["init_wav"] is None
    assert called["init_noise_level"] is None
    assert storage.get(job_id, pipeline.BEAT) == b"generated-beat"


def test_the_plain_class_is_the_one_without_remote_on_it():
    """Stated here so the accessor's reason survives a refactor: the module
    level name is undecorated on purpose — that is what keeps a deploy which
    did not ask for the generator from building its image — and being
    undecorated is exactly what makes it uncallable."""
    from modal_app import beatgen

    with pytest.raises(AttributeError):
        beatgen.BeatGenerator().generate.remote  # noqa: B018


def test_an_upload_source_never_reaches_the_generator(volume_root, monkeypatch, job_store):
    from modal_app import beatgen

    job_id = "c" * 32
    jobs.create(job_id, "beat", params={}, store=job_store)
    monkeypatch.setattr(jobs, "job_dict", job_store)

    def no_gpu():
        raise AssertionError("an uploaded beat paid for a GPU")

    monkeypatch.setattr(beatgen, "generator", no_gpu)
    assert pipeline._generate_beat(job_id, pipeline.clean_params("beat")) is None


def test_only_a_generated_bed_is_re_balanced(volume_root, monkeypatch, job_store):
    """An uploaded beat is somebody's finished production.

    `beats.balance` exists because the generator's output is unmastered by
    construction — a diffusion model has no reason to hand back a balanced
    mix, and the first real job came back with 72% of its power under 120 Hz.
    Running the same shelf over a beat a person mixed would be this pipeline
    deciding it knows better than whoever mixed it.
    """
    from modal_app import beats

    balanced = []
    monkeypatch.setattr(
        beats, "balance", lambda wav, *a, **k: balanced.append(wav) or (b"balanced-bed", "note")
    )
    monkeypatch.setattr(
        beats,
        "analyse_and_fit",
        lambda beat, instrumental, *a, **k: (b"fitted-bed", "plan", "beat-track", "song-track"),
    )
    monkeypatch.setattr(jobs, "job_dict", job_store)

    for index, (source, expected) in enumerate(
        (("upload", b"fitted-bed"), ("generate", b"balanced-bed"), ("derive", b"balanced-bed"))
    ):
        job_id = chr(ord("d") + index) * 32
        jobs.create(job_id, "beat", params={}, store=job_store)
        params = pipeline.clean_params("beat", {"beat_source": source, "beat_prompt": "x"})
        assert pipeline._beat_bed(job_id, params, b"instrumental", b"beat") == expected

    assert balanced == [b"fitted-bed", b"fitted-bed"], "upload was re-balanced, or a source was not"
