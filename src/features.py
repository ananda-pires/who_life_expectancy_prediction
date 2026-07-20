"""
Feature engineering and data preparation utilities.

Pipeline:
1. Summarize data coverage for predictors
2. Remove missing target values
3. Calculate temporal coverage for target and predictors
4. Target missing values removal
5. Impute missing predictor values
6. Assess data coverage
7. Summarize outliers
8. Create engineered features
9. Prepare modelling dataset
10. Select final columns
"""

import numpy as np
import pandas as pd

# =============================================================================
# Summarize data coverage for predictors
# =============================================================================

def summarize_data_coverage(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize variable completeness.

    Returns
    -------
    DataFrame
        Number of available observations,
        percentage coverage,
        and completeness category.
    """

    coverage = pd.DataFrame(
        {
            "Available observations": df.notna().sum()
        }
    )

    coverage["Coverage (%)"] = (
        coverage["Available observations"]
        / len(df)
        * 100
    ).round(2)

    coverage = (
        coverage
        .sort_values(
            "Coverage (%)",
            ascending=False
        )
    )

    coverage["Category"] = pd.cut(
        coverage["Coverage (%)"],
        bins=[0, 25, 50, 75, 100],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ],
        include_lowest=True
    )

    return coverage

# =============================================================================
# Calculate temporal coverage for target and predictors
# =============================================================================

def temporal_coverage(
    df: pd.DataFrame,
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Calculate temporal coverage of all variables.

    Returns
    -------
    DataFrame
        Percentage of available observations
        per year.
    """

    return (
        df
        .groupby(date_column)
        .apply(
            lambda x: x.notna().mean() * 100
        )
        .round(1)
    )


# =============================================================================
# Remove missing values from target feature
# =============================================================================

def remove_missing_target(
    df: pd.DataFrame,
    target: str
) -> pd.DataFrame:
    """
    Remove observations with missing target values.
    """

    return (
        df
        .dropna(subset=[target])
        .reset_index(drop=True)
    )

# =============================================================================
# Median imputation on missing values for predictors
# =============================================================================

def median_imputation(
    df: pd.DataFrame,
    predictors: list[str]
) -> pd.DataFrame:
    """
    Impute missing predictor values using the median.
    """

    df = df.copy()

    df[predictors] = (
        df[predictors]
        .fillna(df[predictors].median())
    )

    return df


# =============================================================================
# Summarize outliers based on IQR method
# =============================================================================

def summarize_outliers(
    df: pd.DataFrame,
    predictors: list[str]
) -> pd.DataFrame:
    """
    Summarize IQR-based outliers.
    """

    summary = []

    for col in predictors:

        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n = (
            (df[col] < lower)
            |
            (df[col] > upper)
        ).sum()

        summary.append(
            {
                "Variable": col,
                "Lower limit": round(lower,2),
                "Upper limit": round(upper,2),
                "Outliers": int(n)
            }
        )

    return pd.DataFrame(summary)
    
# =============================================================================
# Create engineered features
# =============================================================================

def add_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    """
    Create engineered features.
    """

    df = df.copy()

    df["education_index"] = (
        df["mean_years_schooling"]
        +
        df["expected_years_schooling"]
    ) / 2

    df["healthcare_index"] = (
        df["diabetes_treatment"]
        +
        df["hypertension_treatment"]
    ) / 2

    return df

# =============================================================================
# Prepare modelling dataset
# =============================================================================

def prepare_model_dataset(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
    start_year: int,
    end_year: int,
    id_columns: list[str] = ["iso3", "date"]
) -> pd.DataFrame:
    """
    Create the modelling dataset.

    Filters the analysis period and keeps identifier
    variables (iso3 and date) for data integrity checks,
    preprocessing and traceability. Identifier variables
    should be removed before model training.
    """

    return (
        df
        .query("@start_year <= date <= @end_year")
        [
            id_columns
            + predictors
            + [target]
        ]
        .copy()
    )

# =============================================================================
# Select final columns for modelling
# =============================================================================

def select_final_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str
) -> pd.DataFrame:
    """
    Select predictor variables and target variable
    used for model training.
    """

    return (
        df[
            feature_columns + [target_column]
        ]
        .copy()
    )
# =============================================================================
# Define correlation coeficient 
# =============================================================================

def high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.7
) -> pd.Series:
    """
    Return pairs of variables whose absolute Pearson
    correlation exceeds the specified threshold.
    """

    corr = df.corr(numeric_only=True)

    pairs = (
        corr
        .where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        )
        .stack()
    )

    return (
        pairs[
            pairs.abs() >= threshold
        ]
        .sort_values(
            ascending=False
        )
    )