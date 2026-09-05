from fastapi.testclient import TestClient

from modal_app.api import web
from modal_app.app import APP_NAME, DATA_DIR, MODEL_DIR, app


def test_app_identity():
    assert app.name == APP_NAME == "voice-convert"


def test_mount_points():
    assert MODEL_DIR == "/models"
    assert DATA_DIR == "/data"


def test_health_endpoint():
    with TestClient(web) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "voice-convert"
    # It also reports what this deployment can do, so the browser can ask
    # instead of being told at build time — see `test_api`.
    assert "beat_generator" in body


def test_the_formatter_is_pinned_exactly_rather_than_floated():
    """A formatter's output is allowed to change between releases, so a range
    here means CI can go red on a file nobody touched.

    It did: `ruff>=0.6` resolved to 0.16.6 on the runner, 0.16 started
    formatting Python inside Markdown fences, and `ruff format --check` failed
    on a README written under 0.15 — where the same command passed. The lint
    half is fine floating; the format half is not.
    """
    from pathlib import Path

    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    pins = [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip().startswith("ruff")
    ]
    assert pins, "ruff is not in requirements.txt"
    assert len(pins) == 1
    assert pins[0].startswith("ruff=="), f"ruff must be pinned exactly, found {pins[0]!r}"
