"""
Dataset loading utilities for the WHO Health Inequality Data Repository (HIDR).
"""

from pathlib import Path
import requests
import pandas as pd

from src.config import RAW_DATA_DIR


def load_hidr_dataset(dataset_id: str, force_download: bool = False) -> pd.DataFrame:
    """
    Download (if necessary) and load a WHO HIDR dataset.

    Parameters
    ----------
    dataset_id : str
        WHO dataset identifier.
    force_download : bool
        Download again even if a local copy already exists.

    Returns
    -------
    pd.DataFrame
    """

    file_path = RAW_DATA_DIR / f"{dataset_id}.xlsx"

    if force_download or not file_path.exists():

        url = (
            "https://datasafe-h5afbhf4gwctabaa.z01.azurefd.net/"
            f"api/download/TOP/{dataset_id}/data"
        )

        print(f"Downloading {dataset_id}...")

        response = requests.get(url)

        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)

    else:

        print(f"Loading local dataset: {dataset_id}")

    df = pd.read_excel(file_path)

    print(f"{dataset_id}: {df.shape}")

    return df