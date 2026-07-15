"""
Dataset loading and saving utilities for the
WHO Health Inequality Data Repository (HIDR) project.

This module is responsible for:

- Downloading WHO HIDR raw datasets
- Managing local dataset caching
- Loading and saving harmonized analytical datasets
- Loading and saving final machine learning datasets

Keeping data management separated from the analytical workflow
improves reproducibility, maintainability and scalability.
"""


import requests
import pandas as pd


from src.config import (
    RAW_DATA_DIR,
    CACHE_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,
    PROCESSED_DATA_FILE,
    FINAL_DATA_FILE,
)



# =============================================================================
# WHO HIDR raw datasets
# =============================================================================


def load_hidr_dataset(
    dataset_id: str,
    force_download: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download, cache and load a WHO HIDR dataset.

    The first execution downloads the original WHO Excel file.
    A CSV cache is created locally to accelerate future analyses.

    Parameters
    ----------
    dataset_id : str
        WHO HIDR dataset identifier.

    force_download : bool, default=False
        If True, downloads the dataset again even if a local copy exists.

    use_cache : bool, default=True
        If True, saves and loads cached CSV files.

    Returns
    -------
    pd.DataFrame
        WHO HIDR dataset.
    """


    # Define local file paths

    excel_file = (
        RAW_DATA_DIR /
        f"{dataset_id}.xlsx"
    )

    cache_file = (
        CACHE_DIR /
        f"{dataset_id}.csv"
    )


    # Load cached dataset if available

    if (
        use_cache
        and cache_file.exists()
        and not force_download
    ):

        print(
            f"Loading cached dataset: {dataset_id}"
        )

        df = pd.read_csv(
            cache_file,
            encoding="utf-8"
        )

        print(
            f"Shape: {df.shape}"
        )

        return df



    # Download dataset from WHO HIDR API

    if (
        force_download
        or not excel_file.exists()
    ):

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        url = (
            "https://datasafe-h5afbhf4gwctabaa.z01.azurefd.net/"
            f"api/download/TOP/{dataset_id}/data"
        )


        print(
            f"Downloading WHO dataset: {dataset_id}"
        )


        response = requests.get(
            url,
            timeout=60
        )


        response.raise_for_status()


        with open(
            excel_file,
            "wb"
        ) as file:

            file.write(
                response.content
            )


    else:

        print(
            f"Loading local Excel file: {dataset_id}"
        )



    # Read WHO Excel file

    df = pd.read_excel(
        excel_file
    )



    # Create local cache

    if use_cache:

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        df.to_csv(
            cache_file,
            index=False,
            encoding="utf-8"
        )


        print(
            f"Cache created: {cache_file.name}"
        )



    print(
        f"Shape: {df.shape}"
    )


    return df




# =============================================================================
# Dataset summaries
# =============================================================================


def dataset_summary(
    datasets: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generate a summary table describing downloaded datasets.

    Parameters
    ----------
    datasets : dict
        Dictionary containing dataset names and DataFrames.

    Returns
    -------
    pd.DataFrame
        Summary containing number of rows and columns.
    """


    summary = pd.DataFrame(
        {
            "Dataset": datasets.keys(),

            "Rows": [
                df.shape[0]
                for df in datasets.values()
            ],

            "Columns": [
                df.shape[1]
                for df in datasets.values()
            ],
        }
    )


    return (
        summary
        .sort_values(
            "Rows",
            ascending=False
        )
        .reset_index(drop=True)
    )




# =============================================================================
# Harmonized analytical dataset
# =============================================================================


def save_processed_dataset(
    df: pd.DataFrame
) -> None:
    """
    Save the harmonized analytical dataset.

    The processed dataset represents the country-year
    analytical structure generated after extraction,
    harmonization, reshaping and integration.
    """


    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
        encoding="utf-8"
    )


    print(
        "Processed dataset saved:"
    )

    print(
        PROCESSED_DATA_FILE
    )



def load_processed_dataset() -> pd.DataFrame:
    """
    Load the harmonized analytical dataset.

    Returns
    -------
    pd.DataFrame
        Country-year analytical dataset used for EDA.
    """


    if not PROCESSED_DATA_FILE.exists():

        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run harmonization before loading."
        )


    df = pd.read_csv(
        PROCESSED_DATA_FILE,
        encoding="utf-8"
    )


    print(
        f"Loaded processed dataset: {df.shape}"
    )


    return df

# =============================================================================
# Processed analytical dataset
# =============================================================================

from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def save_processed_dataset(
    df,
    filename="model_df_clean.csv"
):
    """
    Save processed dataset.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PROCESSED_DIR / filename

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Processed dataset saved to: {output_path}")

# =============================================================================
# Final machine learning dataset
# =============================================================================


def save_final_dataset(
    df: pd.DataFrame
) -> None:
    """
    Save the final machine learning dataset.

    This dataset contains the processed features
    ready for model training.
    """


    FINAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        FINAL_DATA_FILE,
        index=False,
        encoding="utf-8"
    )


    print(
        "Final ML dataset saved:"
    )

    print(
        FINAL_DATA_FILE
    )




def load_final_dataset() -> pd.DataFrame:
    """
    Load the final machine learning dataset.

    Returns
    -------
    pd.DataFrame
        Dataset ready for predictive modelling.
    """


    if not FINAL_DATA_FILE.exists():

        raise FileNotFoundError(
            "Final ML dataset not found. "
            "Run preprocessing before modelling."
        )


    df = pd.read_csv(
        FINAL_DATA_FILE,
        encoding="utf-8"
    )


    print(
        f"Loaded final ML dataset: {df.shape}"
    )


    return df