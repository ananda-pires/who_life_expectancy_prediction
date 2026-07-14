"""
Feature engineering and data preparation utilities
(Phases 2, 3 and 4).
"""

import pandas as pd

def remove_missing_target(df, target):
    return df.dropna(subset=[target]).reset_index(drop=True)



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