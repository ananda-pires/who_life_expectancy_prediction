"""
Visualization utilities for the WHO Health Inequality Data Repository (HIDR)
life expectancy prediction project.

This module contains reusable plotting functions used during:

- Dataset exploration
- Missing data assessment
- Feature analysis
- Model evaluation

All figures are automatically saved in the project
outputs/figures/ directory.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from src.config import FIGURES_DIR



# =============================================================================
# Internal helper
# =============================================================================


def _save_figure(filename: str) -> None:
    """
    Save matplotlib figure in the project output folder.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        FIGURES_DIR / filename,
        bbox_inches="tight",
        dpi=150
    )



# =============================================================================
# Distribution plots
# =============================================================================


def plot_histogram(
    series: pd.Series,
    title: str,
    filename: str,
    bins: int = 50,
) -> None:
    """
    Plot histogram with KDE curve.

    Used to evaluate variable distributions
    and detect skewness.
    """


    plt.figure(
        figsize=(8,5)
    )


    sns.histplot(
        series.dropna(),
        bins=bins,
        kde=True
    )


    plt.title(title)

    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Missing data visualization
# =============================================================================


def plot_missingness(
    missing_df: pd.DataFrame,
    filename: str = "missing_values.png",
) -> None:
    """
    Plot percentage of missing values by variable.
    """


    plot_data = (
        missing_df
        .sort_values(
            "Percentage (%)"
        )
    )


    plt.figure(
        figsize=(10,6)
    )


    plt.barh(
        plot_data.index,
        plot_data["Percentage (%)"]
    )


    plt.xlabel(
        "Missing values (%)"
    )


    plt.title(
        "Missing data distribution by indicator"
    )


    plt.xlim(
        0,
        100
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Coverage visualization
# =============================================================================


def plot_indicator_coverage(
    coverage: pd.DataFrame,
    filename: str = "indicator_coverage.png",
) -> None:
    """
    Plot availability percentage for analytical variables.
    """


    data = coverage.copy()


    plt.figure(
        figsize=(10,6)
    )


    plt.barh(
        data.index,
        data["Coverage (%)"]
    )


    plt.xlabel(
        "Coverage (%)"
    )


    plt.title(
        "Data availability by indicator"
    )


    plt.xlim(
        0,
        100
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Temporal coverage
# =============================================================================


def plot_temporal_coverage(
    coverage_by_year: pd.DataFrame,
    filename: str = "temporal_coverage.png",
) -> None:
    """
    Plot temporal availability of indicators.
    """


    plt.figure(
        figsize=(16,7)
    )


    plt.imshow(
        coverage_by_year.T,
        aspect="auto",
        interpolation="nearest"
    )


    plt.colorbar(
        label="Coverage (%)"
    )


    plt.yticks(
        range(len(coverage_by_year.columns)),
        coverage_by_year.columns
    )


    plt.xticks(
        range(len(coverage_by_year.index)),
        coverage_by_year.index,
        rotation=90
    )


    plt.xlabel(
        "Year"
    )


    plt.ylabel(
        "Indicator"
    )


    plt.title(
        "Temporal coverage of analytical indicators"
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Correlation analysis
# =============================================================================


def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str = "correlation_heatmap.png",
) -> None:
    """
    Plot Pearson correlation matrix for numerical variables.
    """


    corr = (
        df
        .select_dtypes(
            include=np.number
        )
        .corr()
    )


    plt.figure(
        figsize=(12,10)
    )


    sns.heatmap(
        corr,
        cmap="RdYlGn",
        center=0
    )


    plt.title(
        "Pearson correlation between numerical variables"
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Scatter plots
# =============================================================================


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    filename: str,
) -> None:
    """
    Plot relationship between predictor and target variable.
    """


    plt.figure(
        figsize=(7,5)
    )


    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        alpha=0.3
    )


    plt.title(
        f"{x} vs {y}"
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



# =============================================================================
# Model evaluation plots
# =============================================================================


def plot_observed_vs_predicted(
    y_true,
    y_pred,
    filename: str,
) -> None:
    """
    Plot observed versus predicted values.
    """


    plt.figure(
        figsize=(7,6)
    )


    plt.scatter(
        y_true,
        y_pred,
        alpha=0.3
    )


    limits = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]


    plt.plot(
        limits,
        limits,
        "--"
    )


    plt.xlabel(
        "Observed"
    )


    plt.ylabel(
        "Predicted"
    )


    plt.title(
        "Observed vs predicted values"
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()



def plot_residuals(
    y_true,
    y_pred,
    filename: str,
):
    """
    Plot residual distribution.
    """


    residuals = y_true - y_pred


    plt.figure(
        figsize=(8,5)
    )


    plt.scatter(
        y_pred,
        residuals,
        alpha=0.4
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.xlabel(
        "Predicted values"
    )


    plt.ylabel(
        "Residuals"
    )


    plt.title(
        "Residual analysis"
    )


    plt.tight_layout()


    _save_figure(
        filename
    )


    plt.show()


    return residuals