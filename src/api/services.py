"""Model loading and inference services for the prediction API."""

import json
import logging
import math
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger("life_expectancy_api")

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
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "v2"
MODEL_FILE = MODEL_DIR / "production_pipeline.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"


def load_model_artifacts():
    """Load the complete preprocessing and prediction pipeline from disk."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


def read_model_name() -> str | None:
    """Name of the production model, read from the training metadata."""
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))["production_model"]
    except Exception:
        logger.warning("Could not read model name from %s", METADATA_FILE)
        return None


def predict_life_expectancy(model, features: dict[str, float]) -> float:
    """Return a prediction from the already loaded pipeline."""
    input_frame = pd.DataFrame([features], columns=FEATURES)
    prediction = float(model.predict(input_frame)[0])
    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return prediction
