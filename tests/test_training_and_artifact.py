"""Artifact compatibility and training helper tests."""

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.api import services
from src.modeling.evaluation import (
    FEATURES,
    forward_chaining_splits,
    make_pipeline,
    regression_metrics,
    temporal_train_test_masks,
)


def test_artifact_loads_predicts_and_matches_api_features():
    model = joblib.load(services.MODEL_FILE)
    assert list(model.feature_names_in_) == FEATURES == services.FEATURES
    assert [n for n, _ in model.steps] == ["imputer", "scaler", "model"]
    value = services.predict_life_expectancy(model, dict(zip(FEATURES, [25, 60, 10, 45, 8.5, 12, 6])))
    assert np.isfinite(value) and 30 < value < 100


def test_metadata_and_metrics_report_are_consistent():
    meta = json.loads(services.METADATA_FILE.read_text(encoding="utf-8"))
    report = json.loads((services.MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert meta["features"] == FEATURES
    assert meta["main_protocol"] == report["main_protocol"] == "temporal"
    assert set(report["temporal_metrics"]["Test (2018-2022)"]) == {"MAE", "MSE", "RMSE", "R2"}


def test_temporal_split_never_uses_future_data():
    years = np.repeat(np.arange(2000, 2023), 4)
    train, test = temporal_train_test_masks(years)
    assert years[train].max() < years[test].min()
    for a, b in forward_chaining_splits(years[train]):
        assert years[train][a].max() < years[train][b].min()


def test_pipeline_imputes_and_metrics_run():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 7))
    y = X[:, 0] * 2 + 70
    X[0, 0] = np.nan
    model = make_pipeline(LinearRegression()).fit(X, y)
    m = regression_metrics(y[1:], model.predict(X[1:]))
    assert set(m) == {"MAE", "MSE", "RMSE", "R2"} and m["R2"] > 0.9


def test_imputer_is_fitted_on_training_data_only():
    X_train = np.array([[1.0] * 7, [3.0] * 7, [5.0] * 7, [7.0] * 7])
    pipe = make_pipeline(LinearRegression()).fit(X_train, np.arange(4.0))
    assert pipe.named_steps["imputer"].statistics_[0] == pytest.approx(4.0)
