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
