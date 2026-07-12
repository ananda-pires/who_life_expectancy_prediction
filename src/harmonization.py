"""
Data harmonization utilities for the WHO HIDR project.

This module contains functions for:

- inspecting reporting dimensions;
- aggregating subgroup estimates;
- reshaping datasets from long to wide format;
- merging multiple WHO datasets;
- standardizing variable names;
- generating analytical dataset summaries.

The objective is to transform heterogeneous WHO HIDR datasets
into a unified country-year analytical dataset suitable for
exploratory analysis and machine learning.
"""


import pandas as pd



# =============================================================================
# Inspect dataset dimensions
# =============================================================================


def inspect_reporting_dimensions(
    selected_data: dict
) -> None:
    """
    Inspect demographic and reporting dimensions of selected indicators.

    This step identifies whether indicators are reported by
    sex, age group, residence or other subgroups before aggregation.
    """


    for dataset_name, df in selected_data.items():

        print("=" * 90)
        print(f"Dataset: {dataset_name}")
        print("=" * 90)


        for indicator in sorted(
            df["indicator_name"].unique()
        ):

            print(
                f"\nIndicator: {indicator}"
            )


            summary = (
                df[
                    df["indicator_name"] == indicator
                ]
                .groupby(
                    [
                        "dimension",
                        "subgroup"
                    ],
                    dropna=False
                )
                .size()
                .reset_index(
                    name="observations"
                )
                .sort_values(
                    "observations",
                    ascending=False
                )
            )


            display(summary)



# =============================================================================
# Aggregate subgroup estimates
# =============================================================================


def aggregate_subgroups(
    selected_data: dict
) -> tuple:
    """
    Aggregate subgroup-specific estimates into country-year indicators.

    The WHO HIDR contains indicators reported for different
    population subgroups. To construct a country-year dataset,
    subgroup estimates are averaged.

    Returns
    -------
    aggregated_data : dict
        Aggregated datasets.

    aggregation_summary : pd.DataFrame
        Summary of observations before and after aggregation.
    """


    aggregated_data = {}

    summaries = []


    for dataset_name, df in selected_data.items():


        # Aggregate subgroup estimates
        aggregated = (
            df.groupby(
                [
                    "iso3",
                    "setting",
                    "date",
                    "indicator_name",
                ],
                as_index=False
            )
            .agg(
                estimate=(
                    "estimate",
                    "mean"
                )
            )
        )


        aggregated_data[dataset_name] = aggregated



        # Compare observations before and after aggregation

        before = (
            df.groupby(
                "indicator_name"
            )
            .size()
            .reset_index(
                name="observations_before"
            )
        )


        after = (
            aggregated.groupby(
                "indicator_name"
            )
            .size()
            .reset_index(
                name="observations_after"
            )
        )


        summary = before.merge(
            after,
            on="indicator_name"
        )


        summary["reduction_percent"] = (
            (
                1
                -
                summary["observations_after"]
                /
                summary["observations_before"]
            )
            * 100
        ).round(1)


        summary["dataset"] = dataset_name


        summaries.append(summary)



    aggregation_summary = (
        pd.concat(
            summaries,
            ignore_index=True
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
        aggregation_summary
    )



# =============================================================================
# Reshape datasets
# =============================================================================


def reshape_to_wide(
    aggregated_data: dict
) -> dict:
    """
    Convert datasets from long to wide format.

    Each row becomes a unique country-year observation.
    Each indicator becomes a feature column.
    """


    wide_data = {}


    for dataset_name, df in aggregated_data.items():


        wide = (
            df.pivot(
                index=[
                    "iso3",
                    "setting",
                    "date"
                ],
                columns="indicator_name",
                values="estimate"
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
    wide_data: dict
) -> pd.DataFrame:
    """
    Merge all harmonized datasets into one analytical dataset.

    Uses country, ISO3 and year as common identifiers.
    """


    analytical_dataset = None


    for dataset_name, df in wide_data.items():


        if analytical_dataset is None:

            analytical_dataset = df.copy()


        else:

            analytical_dataset = analytical_dataset.merge(
                df,
                on=[
                    "iso3",
                    "setting",
                    "date"
                ],
                how="outer"
            )


    return analytical_dataset



# =============================================================================
# Standardize variable names
# =============================================================================


def rename_variables(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Rename WHO indicator names into readable snake_case variables.
    """


    column_mapping = {

        # Target
        "Life expectancy (years) ":
            "life_expectancy",


        # Education
        "Expected years of schooling (children aged 6)":
            "expected_years_schooling",

        "Mean years of schooling (population aged 25+)":
            "mean_years_schooling",


        # Poverty
        "Multidimensional Poverty Index":
            "mpi",


        # Environment
        "Concentrations of fine particulate matter (PM2.5)":
            "pm25",

        "Population with primary reliance on polluting fuels and technologies for cooking (%)":
            "polluting_cooking_fuels",


        # WASH
        "Population using safely managed drinking water services (%)":
            "safe_drinking_water",

        "Population using safely managed sanitation services (%)":
            "safe_sanitation",


        # Alcohol
        "Alcohol, consumers in past 12 months (age-standardized) (%)":
            "alcohol_consumers",

        "Alcohol, per capita consumption (15+ years, among drinkers only) (in litres of pure alcohol)":
            "alcohol_consumption",


        # Tobacco
        "Tobacco, current tobacco use prevalence (model-based estimates, age-standardized) (%)":
            "tobacco_use",


        # NCD
        "Obesity prevalence among adults, BMI>=30 (age-standardized) (%)":
            "obesity_prevalence",


        # Healthcare
        "Medical doctors (%)":
            "medical_doctors",

        "Diabetes treatment coverage (30+ years) (age-standardized) (%)":
            "diabetes_treatment",

        "Hypertension treatment coverage among adults aged 30-79 with hypertension (age-standardized) (%)":
            "hypertension_treatment",
    }


    return df.rename(
        columns=column_mapping
    )


# =============================================================================
# Dataset quality summary
# =============================================================================


def dataset_quality_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate basic analytical dataset quality information.
    """


    summary = pd.DataFrame(
        {
            "rows": [
                df.shape[0]
            ],

            "columns": [
                df.shape[1]
            ],

            "countries": [
                df["iso3"].nunique()
            ],

            "years": [
                df["date"].nunique()
            ],
        }
    )


    return summary