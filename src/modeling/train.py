
"""
Model training, persistence, and evaluation utilities for the WHO Health
Inequality Data Repository (HIDR) project.

This module centralizes functions for:

- training machine learning models;
- generating predictions;
- saving and loading trained models;
- saving and loading evaluation metrics;
- versioning model artefacts.

Separating model management from the notebook improves reproducibility,
maintainability, and code reuse.
"""

# =============================================================================
# Imports
# =============================================================================

import json
import joblib

from src.config import (
    MODEL_FILE,
    SCALER_FILE,
    METRICS_FILE,
)

# =============================================================================
# Model training
# =============================================================================

# =============================================================================
# Model prediction
# =============================================================================

# =============================================================================
# Model persistence
# =============================================================================

# =============================================================================
# Evaluation metrics
# =============================================================================


# =============================================================================
# Save trained model
# =============================================================================

def save_model(
    model,
    scaler,
):
    """
    Save the trained model and preprocessing scaler.
    """

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")


# =============================================================================
# Load trained model
# =============================================================================

def load_model():
    """
    Load the trained model and preprocessing scaler.
    """

    model = joblib.load(
        MODEL_FILE,
    )

    scaler = joblib.load(
        SCALER_FILE,
    )

    return model, scaler


# =============================================================================
# Save evaluation metrics
# =============================================================================

def save_metrics(
    metrics,
):
    """
    Save model metadata and evaluation metrics.
    """

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"Metrics saved to: {METRICS_FILE}")


# =============================================================================
# Load evaluation metrics
# =============================================================================

def load_metrics():
    """
    Load model metadata and evaluation metrics.
    """

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)
    
# =============================================================================
# Model version information
# =============================================================================