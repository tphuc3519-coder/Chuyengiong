"""What a deploy actually publishes.

Registration is a side effect of import, which makes it easy to ship a
deployment that is quietly missing the cron or the GPU class. These tests fail
if that happens again.
"""

import importlib

from modal_app import storage
from modal_app.deploy import app


def test_deploy_registers_everything_the_pipeline_needs():
    assert {
        "api",
        "cleanup",
        "run_song_pipeline",
        "run_beat_pipeline",
        "run_rebeat_pipeline",
        "run_vocal_pipeline",
        "run_speech_pipeline",
        "run_tts_pipeline",
        # The operator tools. Nothing in `api.py` calls either, which is
        # exactly why they would otherwise be easy to leave out of a deploy and
        # discover the next time somebody tried to train a voice.
        "train_voice",
        "list_voices",
    } <= set(app.registered_functions)
    assert {
        "VoiceConverter",
        "Separator",
        "Synthesizer",
        "KokoroSynthesizer",
        "OpenJTalkSynthesizer",
        "Watermarker",
    } <= set(app.registered_classes)


def test_the_beat_generator_is_not_in_a_deploy_that_did_not_ask_for_it():
    """`modal deploy` builds every registered image in one pass, so an image
    that fails to build does not fail its own function — it fails the deploy and
    takes every unrelated change in the same push with it. That is not a
    hypothetical: `beatgen_image` does not build, and three phases of working
    code sat undeployed behind it while the live API answered from an older
    revision.

    So the rule is now structural: an image nobody has watched build is not
    attached to the App unless the deployment switched it on."""
    from modal_app import beatgen

    assert not beatgen.enabled()
    assert "BeatGenerator" not in set(app.registered_classes)


def test_the_beat_generator_registers_when_it_is_switched_on(monkeypatch):
    from modal_app import beatgen

    monkeypatch.setattr(beatgen, "enabled", lambda: True)
    assert beatgen.register().__name__ == "BeatGenerator"
    assert "BeatGenerator" in set(app.registered_classes)


def test_every_engine_a_language_reads_through_is_actually_deployed():
    """A language pointing at a class the deploy never registered is a job that
    fails on the first request rather than at deploy time."""
    from modal_app import tts

    for spec in tts.LANGUAGES.values():
        assert tts.ENGINES[spec.engine].__name__ in set(app.registered_classes), spec.label


def test_every_job_mode_has_a_pipeline_that_is_deployed():
    """A mode in the list with no registered function behind it is a job that
    is accepted by `/submit` and then never runs."""
    from modal_app import jobs, pipeline

    for mode in jobs.JOB_MODES:
        assert pipeline.PIPELINES[mode].info.function_name in set(app.registered_functions)


def test_every_config_key_is_actually_passed_by_the_deploy_workflow():
    """`config_secret()` forwards whatever the *deploy machine* has in its
    environment, and the deploy machine is a GitHub Actions runner. A key added
    to `CONFIG_KEYS` but not to the workflow is therefore always empty — the
    feature behind it is simply off, everywhere, with nothing to say so.

    That has now happened twice, so it is a test rather than a habit."""
    from pathlib import Path

    from modal_app.app import CONFIG_KEYS

    workflow = Path(".github/workflows/deploy-modal.yml").read_text()
    for key in CONFIG_KEYS:
        assert f"{key}: " in workflow, f"{key} is in CONFIG_KEYS and not in the deploy workflow"


def test_the_cleanup_cron_sweeps_at_least_as_often_as_the_ttl():
    """Modal does not expose the schedule on a registered Function, so assert on
    the constant the decorator is given."""
    assert storage.CLEANUP_PERIOD_HOURS <= storage.DEFAULT_MAX_AGE_HOURS


def test_importing_the_api_alone_is_not_a_complete_deploy():
    """Why `deploy.py` exists: `modal deploy -m modal_app.api` would leave the
    GPU conversion class out of the deployment."""
    api = importlib.import_module("modal_app.api")
    assert api.api.info.function_name == "api"
    # `conversion` is imported inside the pipeline function bodies, so it is
    # never imported by the web container — and never registered by it either.
    assert not hasattr(api, "VoiceConverter")


def test_the_web_container_does_not_import_the_gpu_stack():
    """The API image has no torch and no audio-separator. Importing either at
    module scope would turn every cold start into an ImportError."""
    for module in ("modal_app.api", "modal_app.pipeline"):
        source = importlib.import_module(module).__file__
        with open(source) as handle:
            top_level = [
                line
                for line in handle
                if line.startswith(("import ", "from ")) and "conversion" in line
            ]
        assert top_level == [], module


def test_everything_a_pipeline_calls_remote_on_is_a_modal_object():
    """The class of bug, not just the instance of it.

    `BeatGenerator().generate.remote(...)` reads exactly like the four calls
    beside it and was not one: that class is undecorated at module scope — on
    purpose, so a deploy that did not ask for the generator never builds its
    image — and `app.cls()` returns a *new* object rather than decorating that
    one. The name in the module is therefore a plain class, and `.remote` on
    its bound method is an attribute that never existed.

    Nothing about the call site says so. This does: every GPU entry point the
    pipeline reaches for has to be something Modal made, and `BeatGenerator` is
    excluded here by name because it is reached through `beatgen.generator()`
    instead — the accessor that exists for this reason.
    """
    import modal

    from modal_app import conversion, separation, tts, watermark

    for name, obj in (
        ("Separator", separation.Separator),
        ("VoiceConverter", conversion.VoiceConverter),
        ("Watermarker", watermark.Watermarker),
        *((f"ENGINES[{key!r}]", engine) for key, engine in tts.ENGINES.items()),
    ):
        assert isinstance(obj, modal.Cls), f"{name} is a {type(obj).__name__}, not a modal.Cls"


def test_the_generators_accessor_is_the_only_way_the_pipeline_names_it():
    """`_generate_beat` must not import the plain class back in.

    Read as source rather than behaviour because that is the mistake worth
    preventing: the working call and the broken one differ by one import line,
    and both compile.
    """
    import inspect

    from modal_app import pipeline

    source = inspect.getsource(pipeline._generate_beat)
    assert "beatgen.generator()" in source
    assert "import BeatGenerator" not in source
