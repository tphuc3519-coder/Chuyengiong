"""What a deploy actually publishes.

Registration is a side effect of import, which makes it easy to ship a
deployment that is quietly missing the cron or the GPU class. These tests fail
if that happens again.
"""

import importlib

from modal_app import storage
from modal_app.deploy import app


def test_deploy_registers_everything_the_pipeline_needs():
    assert {"api", "cleanup"} <= set(app.registered_functions)
    assert "VoiceConverter" in app.registered_classes


def test_the_cleanup_cron_sweeps_at_least_as_often_as_the_ttl():
    """Modal does not expose the schedule on a registered Function, so assert on
    the constant the decorator is given."""
    assert storage.CLEANUP_PERIOD_HOURS <= storage.DEFAULT_MAX_AGE_HOURS


def test_importing_the_api_alone_is_not_a_complete_deploy():
    """Why `deploy.py` exists: `modal deploy -m modal_app.api` publishes one function."""
    api = importlib.import_module("modal_app.api")
    assert api.api.info.function_name == "api"
    # The class lives in a module `api.py` never imports.
    assert not hasattr(api, "VoiceConverter")
