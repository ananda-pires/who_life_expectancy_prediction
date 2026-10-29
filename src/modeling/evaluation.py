"""Helper functions for the temporal life expectancy evaluation protocol.

The training workflow itself (split, validation, model selection, test and
serialization) is executed step by step in notebooks/life_expectancy.ipynb.
This module only holds small reusable pieces shared by the notebook and tests.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Predictors, in the exact order expected by the API.
FEATURES = [
    "obesity_prevalence",
    "diabetes_treatment",
    "polluting_cooking_fuels",
    "hypertension_treatment",
    "mean_years_schooling",
    "expected_years_schooling",
    "alcohol_consumption",
]
TARGET = "life_expectancy"

# Identifiers: used only to structure the splits, never as predictors.
COUNTRY = "iso3"
YEAR = "date"

RANDOM_STATE = 42
START_YEAR = 2000
END_YEAR = 2022

# Main protocol: train on years < TEMPORAL_TEST_START, test on years >= it.
TEMPORAL_TEST_START = 2018

# First year of each forward-chaining validation window (2 years each).
VALIDATION_START_YEARS = (2010, 2012, 2014, 2016)
VALIDATION_WINDOW = 2


def make_pipeline(estimator):
    """Imputation + scaling + estimator in one object.

    Because everything lives in one Pipeline, the imputer and scaler are
    fitted only on the training part of each split or fold.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def regression_metrics(y_true, y_pred):
    """MAE and RMSE in years, MSE in years squared, R2 unitless."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def temporal_train_test_masks(years, test_start=TEMPORAL_TEST_START):
    """Boolean masks (train, test) based on the year value, not row order."""
    years = np.asarray(years)
    train_mask = years < test_start
    return train_mask, ~train_mask


def forward_chaining_splits(
    years,
    start_years=VALIDATION_START_YEARS,
    window=VALIDATION_WINDOW,
):
    """Expanding-window validation splits.

    For each start year S: train on years < S, validate on [S, S + window).
    Validation years always come after the training years, so no future
    information is used. `years` must contain only the training period.
    """
    years = np.asarray(years)
    splits = []
    for start in start_years:
        train_idx = np.flatnonzero(years < start)
        valid_idx = np.flatnonzero((years >= start) & (years < start + window))
        if len(train_idx) and len(valid_idx):
            splits.append((train_idx, valid_idx))
    if len(splits) < 3:
        raise ValueError("Insufficient years for temporal cross-validation.")
    return splits
