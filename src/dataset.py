"""
Utilities for downloading, caching, loading, and saving datasets used in the
WHO Health Inequality Data Repository (HIDR) project.

This module centralizes all data management operations, including:

- Listing available WHO HIDR datasets.
- Downloading datasets from the WHO repository.
- Managing local cache files.
- Summarizing downloaded datasets.
- Loading and saving processed datasets.
- Loading and saving final machine learning datasets.

Separating data management from the analytical workflow improves
reproducibility, maintainability, and code reuse.
"""

from __future__ import annotations

import pandas as pd
import requests

from src.config import (
    CACHE_DIR,
    FINAL_DATA_DIR,
    FINAL_DATA_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
)

# =============================================================================
# WHO HIDR configuration
# =============================================================================

WHO_DOWNLOAD_URL = (
    "https://datasafe-h5afbhf4gwctabaa.z01.azurefd.net/"
    "api/download/TOP"
)

# =============================================================================
# WHO HIDR dataset catalog
# =============================================================================

AVAILABLE_DATASETS = {

    # -------------------------------------------------------------------------
    # COVID-19
    # -------------------------------------------------------------------------

    "rep_covid_cfr": {
        "dataset_name": "COVID-19 case fatality ratios",
        "topic": "COVID-19",
    },

    "rep_covid_rate": {
        "dataset_name": "COVID-19 cases and deaths",
        "topic": "COVID-19",
    },

    "rep_ctis1": {
        "dataset_name": "COVID-19 related mental health and financial worry",
        "topic": "COVID-19",
    },

    "rep_ctis2": {
        "dataset_name": "COVID-19 vaccination",
        "topic": "COVID-19",
    },

    "rep_ctis3": {
        "dataset_name": "COVID-19 burden, behaviours and testing",
        "topic": "COVID-19",
    },

    # -------------------------------------------------------------------------
    # Adult health
    # -------------------------------------------------------------------------

    "rep_dhs_ahn": {
        "dataset_name": "Adult health and nutrition",
        "topic": "Adult health",
    },

    "rep_gho_alcohol": {
        "dataset_name": "Alcohol",
        "topic": "Adult health",
    },

    "rep_gho_ncd": {
        "dataset_name": "Noncommunicable diseases and risk factors",
        "topic": "Adult health",
    },

    "rep_gho_tobacco": {
        "dataset_name": "Tobacco",
        "topic": "Adult health",
    },

    "rep_oecd_ah": {
        "dataset_name": "Adult health",
        "topic": "Adult health",
    },

    # -------------------------------------------------------------------------
    # Health care
    # -------------------------------------------------------------------------

    "rep_dhs_hca": {
        "dataset_name": "Health care access",
        "topic": "Health care",
    },

    "rep_gho_hc": {
        "dataset_name": "Health care system and access",
        "topic": "Health care",
    },

    "rep_eurostat_hc": {
        "dataset_name": "Health care",
        "topic": "Health care",
    },

    "rep_oecd_hc": {
        "dataset_name": "Health care quality, resources and expenditure",
        "topic": "Health care",
    },

    # -------------------------------------------------------------------------
    # Environmental health
    # -------------------------------------------------------------------------

    "rep_gho_env": {
        "dataset_name": "Environmental health",
        "topic": "Environmental health",
    },

    "rep_wash": {
        "dataset_name": "Water, sanitation and hygiene",
        "topic": "Environmental health",
    },

    # -------------------------------------------------------------------------
    # Beyond the health sector
    # -------------------------------------------------------------------------

    "rep_gdl1": {
        "dataset_name": "Development indices",
        "topic": "Beyond the health sector",
    },

    "rep_gdl2": {
        "dataset_name": "Development indicators",
        "topic": "Beyond the health sector",
    },

    "rep_mpi": {
        "dataset_name": "Multidimensional Poverty Index",
        "topic": "Beyond the health sector",
    },

    "rep_wb": {
        "dataset_name": "Health determinants (World Bank)",
        "topic": "Beyond the health sector",
    },

    # -------------------------------------------------------------------------
    # Burden of disease
    # -------------------------------------------------------------------------

    "rep_gho_mortality": {
        "dataset_name": "Life expectancy and mortality",
        "topic": "Burden of disease",
    },

    "rep_ghe_daly_age": {
        "dataset_name": "DALYs by age",
        "topic": "Burden of disease",
    },

    "rep_ghe_daly_sex": {
        "dataset_name": "DALYs by sex",
        "topic": "Burden of disease",
    },

    "rep_ghe_deaths_age": {
        "dataset_name": "Mortality rates by age",
        "topic": "Burden of disease",
    },

    "rep_ghe_deaths_sex": {
        "dataset_name": "Mortality rates by sex",
        "topic": "Burden of disease",
    },

    "rep_ghe_yld_age": {
        "dataset_name": "Years lived with disability",
        "topic": "Burden of disease",
    },

    "rep_ghe_yld_sex": {
        "dataset_name": "Years lived with disability",
        "topic": "Burden of disease",
    },

    "rep_ghe_yll_age": {
        "dataset_name": "Years of life lost",
        "topic": "Burden of disease",
    },

    "rep_ghe_yll_sex": {
        "dataset_name": "Years of life lost",
        "topic": "Burden of disease",
    },

    "rep_ihme_inc": {
        "dataset_name": "Disease incidence estimates",
        "topic": "Burden of disease",
    },

    "rep_ihme_prev": {
        "dataset_name": "Disease prevalence estimates",
        "topic": "Burden of disease",
    },
}

# =============================================================================
# Dataset catalog
# =============================================================================

def list_hidr_datasets() -> pd.DataFrame:
    """
    Return the catalog of datasets available in the WHO Health Inequality
    Data Repository (HIDR).

    Returns
    -------
    pandas.DataFrame
        Table containing dataset identifiers, names, and topic areas.
    """

    return (
        pd.DataFrame.from_dict(
            AVAILABLE_DATASETS,
            orient="index",
        )
        .rename_axis("dataset_id")
        .reset_index()
        .sort_values("topic")
        .reset_index(drop=True)
    )


# =============================================================================
# Dataset download and cache management
# =============================================================================

def load_hidr_dataset(
    dataset_id: str,
    force_download: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download and load a WHO HIDR dataset.

    During the first execution, the original Excel file is downloaded from
    the WHO repository and stored locally. A CSV cache is then created to
    accelerate future executions.

    Parameters
    ----------
    dataset_id : str
        WHO HIDR dataset identifier.

    force_download : bool, default=False
        Download the dataset again even if a local copy already exists.

    use_cache : bool, default=True
        Load and save cached CSV files whenever available.

    Returns
    -------
    pandas.DataFrame
        Requested WHO HIDR dataset.
    """

    excel_file = RAW_DATA_DIR / f"{dataset_id}.xlsx"
    cache_file = CACHE_DIR / f"{dataset_id}.csv"

    if use_cache and cache_file.exists() and not force_download:
        print(f"Loading cached dataset: {dataset_id}")
        return pd.read_csv(cache_file)

    if force_download or not excel_file.exists():

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        url = f"{WHO_DOWNLOAD_URL}/{dataset_id}/data"

        print(f"Downloading dataset: {dataset_id}")

        response = requests.get(
            url,
            timeout=60,
        )

        response.raise_for_status()

        with open(excel_file, "wb") as file:
            file.write(response.content)

    else:
        print(f"Loading local Excel file: {dataset_id}")

    df = pd.read_excel(excel_file)

    if use_cache:

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            cache_file,
            index=False,
            encoding="utf-8",
        )

    return df


# =============================================================================
# Dataset summary
# =============================================================================

def dataset_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate a summary of downloaded datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary containing dataset names as keys and DataFrames as values.

    Returns
    -------
    pandas.DataFrame
        Summary table describing the dimensions and approximate memory usage
        of each dataset.
    """

    summary = pd.DataFrame({

        "Dataset": datasets.keys(),

        "Rows": [
            df.shape[0]
            for df in datasets.values()
        ],

        "Columns": [
            df.shape[1]
            for df in datasets.values()
        ],

        "Memory (MB)": [
            round(
                df.memory_usage(deep=True).sum() / 1024**2,
                2,
            )
            for df in datasets.values()
        ],
    })

    return (
        summary
        .sort_values(
            "Rows",
            ascending=False,
        )
        .reset_index(drop=True)
    )

# =============================================================================
# Processed datasets
# =============================================================================

def save_processed_dataset(
    df: pd.DataFrame,
    filename: str = PROCESSED_DATA_FILE.name,
) -> None:
    """
    Save a processed dataset to the project's processed data directory.

    This function is used to persist intermediate analytical datasets,
    including the harmonized country-year dataset and any additional
    processed datasets generated during preprocessing.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to be saved.

    filename : str, default=PROCESSED_DATA_FILE.name
        Output file name.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = PROCESSED_DATA_DIR / filename

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(f"Processed dataset saved: {output_file}")


def load_processed_dataset(
    filename: str = PROCESSED_DATA_FILE.name,
) -> pd.DataFrame:
    """
    Load a processed dataset from the project's processed data directory.

    Parameters
    ----------
    filename : str, default=PROCESSED_DATA_FILE.name
        Dataset file name.

    Returns
    -------
    pandas.DataFrame
        Loaded processed dataset.

    Raises
    ------
    FileNotFoundError
        If the requested dataset does not exist.
    """

    input_file = PROCESSED_DATA_DIR / filename

    if not input_file.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file,
        encoding="utf-8",
    )

    print(f"Loaded processed dataset: {df.shape}")

    return df


# =============================================================================
# Final machine learning dataset
# =============================================================================

def save_final_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Save the final machine learning dataset.

    This dataset contains all preprocessing and feature engineering
    steps required for model training.

    Parameters
    ----------
    df : pandas.DataFrame
        Final modelling dataset.
    """

    FINAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        FINAL_DATA_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Final dataset saved: {FINAL_DATA_FILE}")


def load_final_dataset() -> pd.DataFrame:
    """
    Load the final machine learning dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset ready for model training.

    Raises
    ------
    FileNotFoundError
        If the final dataset has not yet been created.
    """

    if not FINAL_DATA_FILE.exists():

        raise FileNotFoundError(
            "Final machine learning dataset not found.\n"
            "Run the preprocessing pipeline before loading it."
        )

    df = pd.read_csv(
        FINAL_DATA_FILE,
        encoding="utf-8",
    )

    print(f"Loaded final dataset: {df.shape}")

    return df