"""
Project configuration, directory structure, and central file paths.

These constants are shared across notebooks and modules inside src/.

Centralizing paths improves reproducibility and avoids hard-coded
directories throughout the data processing pipeline.
"""

from pathlib import Path


# =============================================================================
# Project root
# =============================================================================

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]


# =============================================================================
# Data directories
# =============================================================================

# Main data directory
DATA_DIR = ROOT_DIR / "data"

# Original datasets downloaded from the WHO HIDR API
RAW_DATA_DIR = DATA_DIR / "raw"

# Cached CSV files created after the first download
CACHE_DIR = DATA_DIR / "cache"

# Harmonized analytical dataset
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Final dataset after preprocessing
FINAL_DATA_DIR = DATA_DIR / "final"


# =============================================================================
# Output directories
# =============================================================================

# Output folder
OUTPUT_DIR = ROOT_DIR / "outputs"

# Figures generated during EDA and model evaluation
FIGURES_DIR = OUTPUT_DIR / "figures"

# Tables generated throughout the analysis
TABLES_DIR = OUTPUT_DIR / "tables"


# =============================================================================
# Model directories
# =============================================================================

# Root directory for trained models
MODELS_DIR = ROOT_DIR / "models"

# Versioned model directory
MODEL_V1_DIR = MODELS_DIR / "v1"


# =============================================================================
# Standard file names
# =============================================================================

# Harmonized analytical dataset
PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "analytical_dataset.csv"
)

# Final dataset after preprocessing
FINAL_DATA_FILE = (
    FINAL_DATA_DIR /
    "final_ml_dataset.csv"
)

# Trained machine learning model
MODEL_FILE = (
    MODEL_V1_DIR /
    "linear_regression.joblib"
)

# Evaluation metrics
METRICS_FILE = (
    MODEL_V1_DIR /
    "metrics.json"
)


# =============================================================================
# Create project directories
# =============================================================================

for folder in [

    RAW_DATA_DIR,
    CACHE_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,

    FIGURES_DIR,
    TABLES_DIR,

    MODELS_DIR,
    MODEL_V1_DIR,

]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )