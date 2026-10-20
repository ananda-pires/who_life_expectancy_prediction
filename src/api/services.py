"""Model loading and inference services for the prediction API."""

import math

import pandas as pd

from src.config import MODEL_FILE, SCALER_FILE
from src.modeling.train import load_model


FEATURES = [
    "obesity_prevalence",
    "diabetes_treatment",
    "polluting_cooking_fuels",
    "hypertension_treatment",
    "mean_years_schooling",
    "expected_years_schooling",
    "alcohol_consumption",
]

MODEL_VERSION = "v1"


def load_model_artifacts():
    """Load the trained estimator and scaler from disk."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_FILE}")

    if not SCALER_FILE.is_file():
        raise FileNotFoundError(f"Scaler artifact not found: {SCALER_FILE}")

    return load_model()


def predict_life_expectancy(model, scaler, features: dict[str, float]) -> float:
    """Scale validated features and return the model prediction."""
    input_frame = pd.DataFrame([features], columns=FEATURES)
    scaled_input = scaler.transform(input_frame)
    prediction = float(model.predict(scaled_input)[0])

    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return prediction
