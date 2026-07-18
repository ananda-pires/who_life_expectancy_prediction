"""
Utilities for harmonizing WHO Health Inequality Data Repository (HIDR)
datasets into a unified country-year analytical dataset.

The harmonization workflow includes:

- inspecting reporting dimensions;
- aggregating subgroup estimates;
- reshaping datasets from long to wide format;
- merging harmonized datasets;
- standardizing variable names;
- summarizing dataset quality.

These utilities transform heterogeneous WHO HIDR datasets into a
consistent analytical dataset suitable for exploratory analysis
and machine learning.
"""

from IPython.display import display
import pandas as pd

from src.config import WHO_COLUMN_MAPPING


# =============================================================================
# Inspect reporting dimensions
# =============================================================================

def inspect_reporting_dimensions(
    selected_data: dict[str, pd.DataFrame],
) -> None:
    """
    Inspect reporting dimensions for each selected indicator.

    WHO HIDR indicators may be reported across multiple demographic or
    geographic subgroups (e.g., sex, age group, residence). This function
    summarizes the reporting structure before subgroup aggregation.

    Parameters
    ----------
    selected_data : dict[str, pandas.DataFrame]
        Dictionary containing the selected WHO datasets.
    """

    for dataset_name, df in selected_data.items():

        print("=" * 90)
        print(f"Dataset: {dataset_name}")
        print("=" * 90)

        for indicator in sorted(df["indicator_name"].unique()):

            print(f"\nIndicator: {indicator}")

            summary = (
                df.loc[
                    df["indicator_name"] == indicator
                ]
                .groupby(
                    [
                        "dimension",
                        "subgroup",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(name="observations")
                .sort_values(
                    "observations",
                    ascending=False,
                )
            )

            display(summary)


# =============================================================================
# Aggregate subgroup estimates
# =============================================================================

def aggregate_subgroups(
    selected_data: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Aggregate subgroup-specific estimates into country-year observations.

    Several WHO HIDR indicators are reported separately for demographic
    or geographic subgroups. For modelling purposes, subgroup estimates
    are averaged to obtain a single observation per country, year,
    and indicator.

    Parameters
    ----------
    selected_data : dict[str, pandas.DataFrame]
        Dictionary containing selected WHO datasets.

    Returns
    -------
    aggregated_data : dict[str, pandas.DataFrame]
        Harmonized datasets after subgroup aggregation.

    aggregation_summary : pandas.DataFrame
        Summary comparing the number of observations before and after
        aggregation.
    """

    aggregated_data = {}
    summaries = []

    grouping_columns = [
        "iso3",
        "setting",
        "date",
        "indicator_name",
    ]

    for dataset_name, df in selected_data.items():

        aggregated = (
            df.groupby(
                grouping_columns,
                as_index=False,
            )
            .agg(
                estimate=(
                    "estimate",
                    "mean",
                )
            )
        )

        aggregated_data[dataset_name] = aggregated

        before = (
            df.groupby("indicator_name")
            .size()
            .reset_index(
                name="observations_before"
            )
        )

        after = (
            aggregated.groupby("indicator_name")
            .size()
            .reset_index(
                name="observations_after"
            )
        )

        summary = before.merge(
            after,
            on="indicator_name",
        )

        summary["reduction_percent"] = (
            (
                1
                - summary["observations_after"]
                / summary["observations_before"]
            )
            * 100
        ).round(1)

        summary.insert(
            0,
            "dataset",
            dataset_name,
        )

        summaries.append(summary)

    aggregation_summary = (
        pd.concat(
            summaries,
            ignore_index=True,
        )
        [
            [
                "dataset",
                "indicator_name",
                "observations_before",
                "observations_after",
                "reduction_percent",
            ]
        ]
    )

    return (
        aggregated_data,
        aggregation_summary,
    )

# =============================================================================
# Reshape datasets
# =============================================================================

def reshape_to_wide(
    aggregated_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Convert harmonized datasets from long to wide format.

    Each output dataset contains one row per country-year observation,
    while each indicator becomes a feature column.

    Parameters
    ----------
    aggregated_data : dict[str, pandas.DataFrame]
        Harmonized datasets after subgroup aggregation.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing datasets in wide format.
    """

    wide_data = {}

    index_columns = [
        "iso3",
        "setting",
        "date",
    ]

    for dataset_name, df in aggregated_data.items():

        wide = (
            df.pivot(
                index=index_columns,
                columns="indicator_name",
                values="estimate",
            )
            .reset_index()
        )

        wide.columns.name = None

        wide_data[dataset_name] = wide

    return wide_data


# =============================================================================
# Merge datasets
# =============================================================================

def merge_datasets(
    wide_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all harmonized datasets into a single analytical dataset.

    Datasets are merged using ISO3 country code, setting and year as
    common identifiers.

    Parameters
    ----------
    wide_data : dict[str, pandas.DataFrame]
        Dictionary containing harmonized datasets.

    Returns
    -------
    pandas.DataFrame
        Integrated analytical dataset.
    """

    analytical_dataset = None

    merge_keys = [
        "iso3",
        "setting",
        "date",
    ]

    for df in wide_data.values():

        if analytical_dataset is None:

            analytical_dataset = df.copy()

        else:

            analytical_dataset = analytical_dataset.merge(
                df,
                on=merge_keys,
                how="outer",
            )

    return analytical_dataset


# =============================================================================
# Standardize variable names
# =============================================================================

def rename_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rename WHO indicator names using standardized project variable names.

    Parameters
    ----------
    df : pandas.DataFrame
        Analytical dataset with original WHO indicator names.

    Returns
    -------
    pandas.DataFrame
        Dataset with standardized snake_case variable names.
    """

    return df.rename(
        columns=WHO_COLUMN_MAPPING
    )


# =============================================================================
# Dataset quality summary
# =============================================================================

def dataset_quality_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a summary describing the analytical dataset.

    The summary provides basic structural information that helps
    verify the dataset after harmonization.

    Parameters
    ----------
    df : pandas.DataFrame
        Harmonized analytical dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset quality summary.
    """

    total_cells = df.shape[0] * df.shape[1]

    missing_values = df.isna().sum().sum()

    summary = pd.DataFrame(
        {
            "Rows": [
                df.shape[0]
            ],
            "Columns": [
                df.shape[1]
            ],
            "Countries": [
                df["iso3"].nunique()
            ],
            "Years": [
                df["date"].nunique()
            ],
            "Missing values": [
                missing_values
            ],
            "Missing (%)": [
                round(
                    missing_values /
                    total_cells *
                    100,
                    2,
                )
            ],
            "Memory (MB)": [
                round(
                    df.memory_usage(
                        deep=True
                    ).sum()
                    / 1024**2,
                    2,
                )
            ],
        }
    )

    return summary