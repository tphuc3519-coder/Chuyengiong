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
        "run_speech_pipeline",
        "run_tts_pipeline",
    } <= set(app.registered_functions)
    assert {
        "VoiceConverter",
        "Separator",
        "Synthesizer",
        "KokoroSynthesizer",
        "OpenJTalkSynthesizer",
        "Watermarker",
    } <= set(app.registered_classes)


def test_every_engine_a_language_reads_through_is_actually_deployed():
    """A language pointing at a class the deploy never registered is a job that
    fails on the first request rather than at deploy time."""
    from modal_app import tts

    for spec in tts.LANGUAGES.values():
        assert tts.ENGINES[spec.engine].__name__ in set(app.registered_classes), spec.label


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
