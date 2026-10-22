"""Model loading and inference services for the prediction API."""

import math
from pathlib import Path

import joblib
import pandas as pd


FEATURES = [
    "obesity_prevalence",
    "diabetes_treatment",
    "polluting_cooking_fuels",
    "hypertension_treatment",
    "mean_years_schooling",
    "expected_years_schooling",
    "alcohol_consumption",
]

MODEL_VERSION = "v2"
MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "v2"
    / "production_pipeline.joblib"
)


def load_model_artifacts():
    """Load the complete preprocessing and prediction pipeline."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_FILE}")

    return joblib.load(MODEL_FILE)


def predict_life_expectancy(model, features: dict[str, float]) -> float:
    """Return a prediction from the complete trained pipeline."""
    input_frame = pd.DataFrame([features], columns=FEATURES)
    prediction = float(model.predict(input_frame)[0])

    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return prediction
