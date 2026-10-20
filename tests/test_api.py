"""Tests for the life expectancy prediction API."""

from fastapi.testclient import TestClient

from src.api.main import app


VALID_PAYLOAD = {
    "obesity_prevalence": 25.0,
    "diabetes_treatment": 60.0,
    "polluting_cooking_fuels": 10.0,
    "hypertension_treatment": 45.0,
    "mean_years_schooling": 8.5,
    "expected_years_schooling": 12.0,
    "alcohol_consumption": 6.0,
}


def test_health_endpoint_reports_loaded_model():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint_returns_valid_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert 0 < body["prediction"] < 150
    assert body["target"] == "life_expectancy"
    assert body["model_version"] == "v1"


def test_prediction_rejects_percentage_above_100():
    invalid_payload = {
        **VALID_PAYLOAD,
        "obesity_prevalence": 125.0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_prediction_rejects_string_for_numeric_field():
    invalid_payload = {
        **VALID_PAYLOAD,
        "obesity_prevalence": "25.0",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_prediction_rejects_missing_field():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("alcohol_consumption")

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_prediction_rejects_unexpected_field():
    invalid_payload = {
        **VALID_PAYLOAD,
        "country": "Example",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422