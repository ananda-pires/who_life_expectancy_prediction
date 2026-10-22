"""Leakage-safe Linear Regression and KNN training and evaluation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR, ROOT_DIR


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
COUNTRY = "iso3"
YEAR = "date"

START_YEAR = 2000
END_YEAR = 2022
TEMPORAL_TEST_START = 2018
RANDOM_STATE = 42

MODEL_DIR = ROOT_DIR / "models" / "v2"
TABLE_DIR = ROOT_DIR / "outputs" / "tables" / "v2"
FIGURE_DIR = ROOT_DIR / "outputs" / "figures" / "v2"

for directory in (MODEL_DIR, TABLE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PARAM_GRID = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}


def make_pipeline(estimator):
    """Fit imputation and scaling only on each training fold."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def tune_knn(X, y, cv, groups=None):
    search = GridSearchCV(
        estimator=make_pipeline(KNeighborsRegressor()),
        param_grid=PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search


def temporal_cv(years):
    """Forward-chaining validation; validation years follow training years."""
    years = np.asarray(years)
    splits = []
    for start in (2010, 2012, 2014, 2016):
        train_idx = np.flatnonzero(years < start)
        valid_idx = np.flatnonzero((years >= start) & (years < start + 2))
        if len(train_idx) and len(valid_idx):
            splits.append((train_idx, valid_idx))
    if len(splits) < 3:
        raise ValueError("Insufficient years for temporal cross-validation.")
    return splits


def save_predictions(metadata, y_true, y_pred, name):
    result = metadata[[COUNTRY, YEAR]].reset_index(drop=True).copy()
    result["observed"] = np.asarray(y_true)
    result["predicted"] = np.asarray(y_pred)
    result["residual"] = result["observed"] - result["predicted"]
    result["absolute_error"] = result["residual"].abs()
    result.to_csv(TABLE_DIR / f"{name}_predictions.csv", index=False)
    return result


def save_diagnostics(predictions, name):
    by_year = predictions.groupby(YEAR).agg(
        observations=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("residual", lambda x: np.sqrt(np.mean(np.square(x)))),
    ).reset_index()
    by_country = predictions.groupby(COUNTRY).agg(
        observations=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("residual", lambda x: np.sqrt(np.mean(np.square(x)))),
    ).reset_index().sort_values("MAE", ascending=False)

    by_year.to_csv(TABLE_DIR / f"{name}_mae_by_year.csv", index=False)
    by_country.to_csv(TABLE_DIR / f"{name}_mae_by_country.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions["observed"], predictions["predicted"], alpha=0.5)
    low = min(predictions["observed"].min(), predictions["predicted"].min())
    high = max(predictions["observed"].max(), predictions["predicted"].max())
    ax.plot([low, high], [low, high], "--")
    ax.set(
        xlabel="Observed life expectancy (years)",
        ylabel="Predicted life expectancy (years)",
        title=f"Observed vs predicted: {name}",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_observed_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predictions["predicted"], predictions["residual"], alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Predicted life expectancy (years)",
        ylabel="Residual (observed - predicted, years)",
        title=f"Residuals: {name}",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_residuals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(by_year[YEAR], by_year["MAE"], marker="o")
    ax.set(xlabel="Year", ylabel="MAE (years)", title=f"MAE by year: {name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_mae_by_year.png", dpi=160)
    plt.close(fig)

    worst = by_country.head(20).sort_values("MAE")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(worst[COUNTRY], worst["MAE"])
    ax.set(
        xlabel="MAE (years)",
        ylabel="Country code",
        title=f"Highest country-level MAE: {name}",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_mae_by_country.png", dpi=160)
    plt.close(fig)


def record_result(rows, protocol, model_name, train_n, test_n,
                  train_countries, test_countries, y_true, y_pred,
                  best_params=None, cv_mae=None):
    rows.append({
        "protocol": protocol,
        "model": model_name,
        "train_observations": int(train_n),
        "test_observations": int(test_n),
        "train_countries": int(train_countries),
        "test_countries": int(test_countries),
        **metrics(y_true, y_pred),
        "cv_MAE": float(cv_mae) if cv_mae is not None else np.nan,
        "best_params": json.dumps(best_params or {}, sort_keys=True),
    })


def main():
    source = PROCESSED_DATA_DIR / "analytical_dataset.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    df = pd.read_csv(source)
    required = [COUNTRY, YEAR, TARGET, *FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in [YEAR, TARGET, *FEATURES]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Do not impute the target. Rows without a known target cannot be trained/evaluated.
    df = df.loc[
        df[YEAR].between(START_YEAR, END_YEAR)
        & df[COUNTRY].notna()
        & df[TARGET].notna()
    ].copy()
    df[YEAR] = df[YEAR].astype(int)
    df = df.sort_values([YEAR, COUNTRY]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable observations remain after filtering.")
    if df.duplicated([COUNTRY, YEAR]).any():
        raise ValueError("Duplicate country-year observations detected.")

    X = df[FEATURES]
    y = df[TARGET]
    groups = df[COUNTRY]

    print(f"Dataset: {source}")
    print(f"Rows: {len(df)} | Countries: {groups.nunique()}")
    print(f"Years: {df[YEAR].min()}-{df[YEAR].max()}")
    print(f"Rows with missing predictors: {int(X.isna().any(axis=1).sum())}")

    results = []

    # EXPERIMENT 1: later years for countries represented in earlier years.
    time_train = df[YEAR] < TEMPORAL_TEST_START
    time_test = ~time_train
    Xtt, ytt = X.loc[time_train].reset_index(drop=True), y.loc[time_train].reset_index(drop=True)
    Xte, yte = X.loc[time_test].reset_index(drop=True), y.loc[time_test].reset_index(drop=True)
    time_cv = temporal_cv(df.loc[time_train, YEAR].to_numpy())

    temporal_knn = tune_knn(Xtt, ytt, time_cv)

    temporal_lr_cv_scores = cross_val_score(
        make_pipeline(LinearRegression()),
        Xtt,
        ytt,
        cv=time_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    temporal_lr_cv_mae = float(-temporal_lr_cv_scores.mean())

    temporal_lr = make_pipeline(LinearRegression())
    temporal_lr.fit(Xtt, ytt)

    train_countries = set(df.loc[time_train, COUNTRY])
    test_countries = set(df.loc[time_test, COUNTRY])
    unseen_temporal_countries = sorted(test_countries - train_countries)

    print(
        "Temporal split country overlap: "
        f"{len(test_countries & train_countries)} of "
        f"{len(test_countries)} test countries also occur in training."
    )
    if unseen_temporal_countries:
        print(
            "Warning: countries in the temporal test set but absent "
            f"from training: {unseen_temporal_countries}"
        )

    time_meta = df.loc[time_test, [COUNTRY, YEAR]].reset_index(drop=True)
    for name, fitted, cv_mae, params in [
        ("linear_regression", temporal_lr, temporal_lr_cv_mae, None),
        ("knn", temporal_knn.best_estimator_, -temporal_knn.best_score_,
         temporal_knn.best_params_),
    ]:
        pred = fitted.predict(Xte)
        saved = save_predictions(time_meta, yte, pred, f"temporal_{name}")
        save_diagnostics(saved, f"temporal_{name}")
        record_result(
            results, "temporal", name, len(Xtt), len(Xte),
            df.loc[time_train, COUNTRY].nunique(),
            df.loc[time_test, COUNTRY].nunique(),
            yte, pred, params, cv_mae,
        )
        joblib.dump(fitted, MODEL_DIR / f"temporal_{name}_pipeline.joblib")

    # EXPERIMENT 2: unseen countries. Split whole countries, not individual rows.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    country_train_idx, country_test_idx = next(
        splitter.split(X, y, groups=groups)
    )
    Xct, yct = X.iloc[country_train_idx].reset_index(drop=True), y.iloc[country_train_idx].reset_index(drop=True)
    Xce, yce = X.iloc[country_test_idx].reset_index(drop=True), y.iloc[country_test_idx].reset_index(drop=True)
    gct = groups.iloc[country_train_idx].reset_index(drop=True)
    country_cv = GroupKFold(n_splits=5)

    country_knn = tune_knn(Xct, yct, country_cv, groups=gct)
    country_lr = make_pipeline(LinearRegression())
    country_lr.fit(Xct, yct)

    # Comparable grouped CV score for the untuned linear baseline.
    lr_cv_scores = cross_val_score(
        make_pipeline(LinearRegression()),
        Xct, yct, groups=gct, cv=country_cv,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    lr_cv_mae = float(-lr_cv_scores.mean())
    knn_cv_mae = float(-country_knn.best_score_)

    country_meta = df.iloc[country_test_idx][[COUNTRY, YEAR]].reset_index(drop=True)
    for name, fitted, cv_mae, params in [
        ("linear_regression", country_lr, lr_cv_mae, None),
        ("knn", country_knn.best_estimator_, knn_cv_mae, country_knn.best_params_),
    ]:
        pred = fitted.predict(Xce)
        saved = save_predictions(country_meta, yce, pred, f"country_holdout_{name}")
        save_diagnostics(saved, f"country_holdout_{name}")
        record_result(
            results, "unseen_countries", name, len(Xct), len(Xce),
            gct.nunique(),
            groups.iloc[country_test_idx].nunique(),
            yce, pred, params, cv_mae,
        )
        joblib.dump(fitted, MODEL_DIR / f"country_holdout_{name}_pipeline.joblib")

    # Select using grouped CV MAE on the country-training data, not holdout scores.
    if knn_cv_mae < lr_cv_mae:
        production_name = "knn"
        production_pipeline = make_pipeline(
            KNeighborsRegressor(**{
                key.replace("model__", ""): value
                for key, value in country_knn.best_params_.items()
            })
        )
        production_cv_mae = knn_cv_mae
    else:
        production_name = "linear_regression"
        production_pipeline = make_pipeline(LinearRegression())
        production_cv_mae = lr_cv_mae

    # Refit the selected pipeline on all eligible labelled data for deployment.
    production_pipeline.fit(X, y)
    joblib.dump(production_pipeline, MODEL_DIR / "production_pipeline.joblib")
    # Also preserve an explicit KNN artifact for inspection and comparison.
    joblib.dump(country_knn.best_estimator_, MODEL_DIR / "knn_pipeline.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLE_DIR / "evaluation_metrics.csv", index=False)
    results_df.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    metadata = {
        "model_version": "v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "random_state": RANDOM_STATE,
        "source_dataset": source.relative_to(ROOT_DIR).as_posix(),
        "target": TARGET,
        "features": FEATURES,
        "period": [START_YEAR, END_YEAR],
        "n_observations": int(len(df)),
        "n_countries": int(groups.nunique()),
        "temporal_test_start_year": TEMPORAL_TEST_START,
        "temporal_knn_best_params": temporal_knn.best_params_,
        "country_knn_best_params": country_knn.best_params_,
        "country_cv_mae": {
            "linear_regression": lr_cv_mae,
            "knn": knn_cv_mae,
        },
        "production_model": production_name,
        "production_selection_metric": "5-fold GroupKFold mean MAE",
        "production_selection_cv_mae": production_cv_mae,
        "production_artifact": "production_pipeline.joblib",
        "note": (
            "Temporal and unseen-country results answer different questions. "
            "The production pipeline is refitted on all labelled observations "
            "after model selection using grouped cross-validation."
        ),
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nEvaluation results:")
    print(results_df[["protocol", "model", "MAE", "RMSE", "R2", "cv_MAE"]].to_string(index=False))
    print(f"\nProduction model selected by grouped CV: {production_name}")
    print(f"Grouped CV MAE: {production_cv_mae:.4f} years")
    print(f"Artifacts: {MODEL_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
