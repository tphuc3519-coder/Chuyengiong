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
    assert response.json() == {"status": "ok", "app": "voice-convert"}
