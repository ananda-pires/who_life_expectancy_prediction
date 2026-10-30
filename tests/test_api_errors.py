"""Lifecycle, error-handling and HTTP status tests (all failures simulated)."""

from unittest.mock import patch

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import services
from src.api.main import app

VALID = {
    "obesity_prevalence": 25.0,
    "diabetes_treatment": 60.0,
    "polluting_cooking_fuels": 10.0,
    "hypertension_treatment": 45.0,
    "mean_years_schooling": 8.5,
    "expected_years_schooling": 12.0,
    "alcohol_consumption": 6.0,
}


def test_model_loaded_once_and_not_reloaded_per_request():
    with patch.object(services.joblib, "load", wraps=joblib.load) as spy:
        with TestClient(app) as client:
            for _ in range(5):
                assert client.post("/predict", json=VALID).status_code == 200
    assert spy.call_count == 1


def test_lifespan_stores_model_and_name():
    import json
    expected = json.loads(services.METADATA_FILE.read_text(encoding="utf-8"))["production_model"]
    with TestClient(app) as client:
        assert client.app.state.model is not None
        assert client.app.state.model_name == expected


def test_health_ok_has_model_name():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_name"]


def test_422_text_in_numeric_field_has_friendly_body():
    with TestClient(app) as client:
        r = client.post("/predict", json={**VALID, "obesity_prevalence": "abc"})
    assert r.status_code == 422
    detail = r.json()["error"]["details"][0]
    assert r.json()["error"]["code"] == "validation_error"
    assert detail["field"] == "obesity_prevalence"
    assert "expected" in detail and "how_to_fix" in detail


@pytest.mark.parametrize("field,value", [
    ("obesity_prevalence", -1.0),
    ("diabetes_treatment", 100.5),
    ("mean_years_schooling", 21.0),
    ("expected_years_schooling", 26.0),
    ("alcohol_consumption", 101.0),
])
def test_422_out_of_range(field, value):
    with TestClient(app) as client:
        r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == field


def test_422_missing_field_and_extra_field():
    missing = {k: v for k, v in VALID.items() if k != "alcohol_consumption"}
    with TestClient(app) as client:
        r1 = client.post("/predict", json=missing)
        r2 = client.post("/predict", json={**VALID, "country": "BRA"})
    assert r1.status_code == 422 and r2.status_code == 422
    assert r1.json()["error"]["details"][0]["field"] == "alcohol_consumption"
    assert r2.json()["error"]["details"][0]["field"] == "country"


def test_400_malformed_json():
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            content='{"obesity_prevalence": 25.0,',
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "malformed_json"


def test_500_simulated_inference_failure_hides_details():
    with TestClient(app, raise_server_exceptions=False) as client:
        with patch("src.api.routers.predict_life_expectancy", side_effect=RuntimeError("secret /path")):
            r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert "secret" not in r.text and "/path" not in r.text


def test_500_unexpected_exception_hides_details():
    with TestClient(app, raise_server_exceptions=False) as client:
        with patch("src.api.routers.predict_life_expectancy", side_effect=KeyError("internal")):
            r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "internal" not in r.json()["error"]["message"]


def test_503_when_model_unavailable_at_runtime():
    with TestClient(app) as client:
        client.app.state.model = None
        assert client.post("/predict", json=VALID).status_code == 503
        h = client.get("/health")
    assert h.status_code == 503
    assert h.json()["model_loaded"] is False


def test_startup_with_missing_artifact_is_degraded_not_ready():
    with patch("src.api.main.load_model_artifacts", side_effect=FileNotFoundError("missing")):
        with TestClient(app) as client:
            h = client.get("/health")
            p = client.post("/predict", json=VALID)
    assert h.status_code == 503 and h.json()["status"] == "unavailable"
    assert p.status_code == 503
