"""
plots.py

Reusable visualization functions used throughout the project.

Each function:
- creates a matplotlib/seaborn figure;
- saves the figure automatically to outputs/figures;
- displays the figure.

The module is used for:
- Exploratory Data Analysis (EDA);
- Feature analysis;
- Model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR



# ======================================================
# Global visualization settings
# ======================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# ======================================================
# Helper function
# ======================================================

def _savefig(filename: str) -> None:
    """
    Save figure to the project figures directory.

    Parameters
    ----------
    filename : str
        Name of the output image file.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )


# ======================================================
# Exploratory Data Analysis (EDA)
# ======================================================


def plot_histogram(
    series: pd.Series,
    title: str,
    filename: str,
    bins: int = 30
) -> None:
    """
    Plot histogram with kernel density estimate (KDE).

    Parameters
    ----------
    series : pd.Series
        Numerical variable to visualize.

    title : str
        Figure title.

    filename : str
        Output filename.

    bins : int, default=30
        Number of histogram bins.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.histplot(
        series.dropna(),
        bins=bins,
        kde=True,
        color="#2E7D32",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    filename: str
) -> None:
    """
    Plot boxplots for numerical variables.

    Useful for identifying potential outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing variables.

    columns : list[str]
        Variables to visualize.

    filename : str
        Output filename.
    """

    fig, axes = plt.subplots(
        ncols=len(columns),
        figsize=(4 * len(columns), 5)
    )

    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):

        sns.boxplot(
            y=df[column],
            color="#81C784",
            ax=ax
        )

        ax.set_title(column)


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_multiple_histograms(
    df: pd.DataFrame,
    columns: list[str],
    filename: str,
    ncols: int = 2
) -> None:
    """
    Plot distributions of multiple numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.

    columns : list[str]
        Variables to plot.

    filename : str
        Output filename.

    ncols : int, default=2
        Number of subplot columns.
    """

    nrows = int(
        np.ceil(len(columns) / ncols)
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4 * nrows)
    )

    axes = np.array(axes).reshape(-1)


    for ax, column in zip(axes, columns):

        sns.histplot(
            df[column],
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {column}"
        )

        ax.set_xlabel(column)


    for ax in axes[len(columns):]:
        ax.remove()


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    filename: str
) -> None:
    """
    Plot relationship between two numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.

    x : str
        Predictor variable.

    y : str
        Target variable.

    filename : str
        Output filename.
    """

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        alpha=0.4,
        s=25,
        color="#1565C0",
        ax=ax
    )

    ax.set_title(
        f"{y} vs {x}"
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str
) -> None:
    """
    Plot Pearson correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numerical variables.

    filename : str
        Output filename.
    """

    correlation = (
        df
        .select_dtypes(include=np.number)
        .corr()
    )


    fig, ax = plt.subplots(
        figsize=(12, 10)
    )


    sns.heatmap(
        correlation,
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax
    )


    ax.set_title(
        "Pearson Correlation Matrix"
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()

# ======================================================
# Data completeness and availability
# ======================================================


def plot_missing_values(
    df: pd.DataFrame,
    filename: str
) -> None:
    """
    Plot percentage of missing values per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Analytical dataset.

    filename : str
        Output filename.
    """

    missing = (
        df
        .isna()
        .mean()
        .sort_values(ascending=False)
        * 100
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.barplot(
        x=missing.index,
        y=missing.values,
        color="#5C6BC0",
        ax=ax
    )


    ax.set_title(
        "Percentage of Missing Values by Variable"
    )

    ax.set_ylabel(
        "Missing values (%)"
    )

    ax.set_xlabel(
        ""
    )


    ax.tick_params(
        axis="x",
        rotation=90
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()


def plot_indicator_coverage(
    coverage: pd.DataFrame,
    filename: str
) -> None:
    """
    Plot indicator data availability grouped by coverage category.

    Parameters
    ----------
    coverage : pd.DataFrame
        DataFrame containing:
        - Coverage (%)
        - Category

    filename : str
        Output filename.
    """

    data = (
        coverage
        .reset_index()
        .rename(columns={"index": "Indicator"})
        .sort_values(
            "Coverage (%)",
            ascending=True
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    sns.barplot(
        data=data,
        x="Coverage (%)",
        y="Indicator",
        hue="Category",
        dodge=False,
        ax=ax
    )


    ax.set_title(
        "Indicator Data Availability"
    )

    ax.set_xlabel(
        "Available observations (%)"
    )

    ax.set_ylabel(
        ""
    )


    ax.legend(
        title="Coverage category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()


def plot_temporal_coverage(
    coverage_by_year: pd.DataFrame,
    filename: str
) -> None:
    """
    Plot temporal availability of indicators.

    Parameters
    ----------
    coverage_by_year : pd.DataFrame
        DataFrame where rows represent years
        and columns represent indicators.

    filename : str
        Output filename.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    sns.heatmap(
        coverage_by_year.T,
        cmap="viridis",
        vmin=0,
        vmax=100,
        ax=ax
    )


    ax.set_title(
        "Temporal Availability of Indicators"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Indicator"
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_country_completeness(
    completeness: pd.DataFrame,
    filename: str,
    top_n: int = 15
) -> None:
    """
    Plot countries with highest and lowest data completeness.

    Parameters
    ----------
    completeness : pd.DataFrame
        Country-level completeness dataframe.

    filename : str
        Output filename.

    top_n : int, default=15
        Number of countries displayed.
    """

    data = completeness.copy()


    data["Overall completeness"] = (
        data
        .select_dtypes(include=np.number)
        .mean(axis=1)
    )


    data = (
        data
        .sort_values(
            "Overall completeness"
        )
        .head(top_n)
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    sns.barplot(
        data=data,
        x="Overall completeness",
        y="setting",
        ax=ax
    )


    ax.set_title(
        "Countries with Lowest Data Completeness"
    )


    ax.set_xlabel(
        "Completeness (%)"
    )


    ax.set_ylabel(
        ""
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str
) -> None:
    """
    Plot Pearson correlation heatmap with correlation coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Numerical variables used for correlation analysis.

    filename : str
        Output filename.
    """

    corr = df.corr()

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    sns.heatmap(
        corr,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Pearson correlation matrix"
    )

    plt.tight_layout()

    _savefig(filename)

    plt.show()
    

# ======================================================
# Time series analysis
# ======================================================


def plot_country_trend(
    df: pd.DataFrame,
    country: str,
    target: str,
    filename: str
) -> None:
    """
    Plot temporal evolution of an indicator for a country.

    Parameters
    ----------
    df : pd.DataFrame
        Analytical dataset.

    country : str
        Country name.

    target : str
        Variable to plot.

    filename : str
        Output filename.
    """

    data = (
        df[df["setting"] == country]
        .sort_values("date")
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    sns.lineplot(
        data=data,
        x="date",
        y=target,
        marker="o",
        linewidth=2,
        ax=ax
    )


    ax.set_title(
        f"{target} trend - {country}"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        target
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_global_trend(
    df: pd.DataFrame,
    target: str,
    filename: str
) -> None:
    """
    Plot global mean trend of a variable over time.

    Parameters
    ----------
    df : pd.DataFrame
        Analytical dataset.

    target : str
        Variable to summarize.

    filename : str
        Output filename.
    """

    trend = (
        df
        .groupby("date")[target]
        .mean()
        .reset_index()
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    sns.lineplot(
        data=trend,
        x="date",
        y=target,
        marker="o",
        linewidth=2,
        ax=ax
    )


    ax.set_title(
        f"Global Average {target} Trend"
    )


    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        target
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()



# ======================================================
# Model evaluation
# ======================================================


def plot_observed_vs_predicted(
    y_true,
    y_pred,
    title: str,
    filename: str
) -> None:
    """
    Plot observed versus predicted values.

    Parameters
    ----------
    y_true : array-like
        Observed target values.

    y_pred : array-like
        Predicted target values.

    title : str
        Figure title.

    filename : str
        Output filename.
    """

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    ax.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=25,
        color="#2E7D32"
    )


    limits = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]


    ax.plot(
        limits,
        limits,
        "--",
        color="#C62828",
        linewidth=1.5,
        label="Perfect prediction"
    )


    ax.set_xlabel(
        "Observed values"
    )

    ax.set_ylabel(
        "Predicted values"
    )

    ax.set_title(
        title
    )


    ax.legend()


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_residuals(
    y_true,
    y_pred,
    title: str,
    filename: str
) -> pd.Series:
    """
    Plot model residuals.

    Residuals are calculated as:
    observed - predicted

    Parameters
    ----------
    y_true : array-like
        Observed target values.

    y_pred : array-like
        Predicted target values.

    title : str
        Figure title.

    filename : str
        Output filename.

    Returns
    -------
    pd.Series
        Calculated residuals.
    """

    residuals = (
        pd.Series(y_true)
        -
        pd.Series(y_pred)
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.scatter(
        y_pred,
        residuals,
        alpha=0.5,
        color="#81C784"
    )


    ax.axhline(
        0,
        linestyle="--",
        color="#C62828",
        linewidth=1.5
    )


    ax.set_xlabel(
        "Predicted values"
    )

    ax.set_ylabel(
        "Residuals"
    )

    ax.set_title(
        title
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()


    return residuals



def plot_feature_importance(
    importance,
    feature_names,
    filename: str,
    top_n: int = 20
) -> None:
    """
    Plot feature importance from a machine learning model.

    Parameters
    ----------
    importance : array-like
        Importance values.

    feature_names : list
        Predictor names.

    filename : str
        Output filename.

    top_n : int, default=20
        Number of features displayed.
    """

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance
            }
        )
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )


    ax.set_title(
        "Feature Importance"
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()



def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str,
    filename: str
) -> None:
    """
    Plot comparison between machine learning models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing model performance.

    metric : str
        Metric used for comparison.

    filename : str
        Output filename.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    sns.barplot(
        data=metrics_df,
        x="Model",
        y=metric,
        ax=ax
    )


    ax.set_title(
        f"Model Comparison - {metric}"
    )


    ax.tick_params(
        axis="x",
        rotation=45
    )


    plt.tight_layout()

    _savefig(filename)

    plt.show()