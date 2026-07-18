"""
Project configuration for the WHO Health Inequality Data Repository (HIDR)
machine learning project.

This module centralizes the project's directory structure, standard file paths,
and project-wide constants used throughout the data processing pipeline.

Keeping all configuration in a single location improves reproducibility,
maintainability, and portability.
"""

from pathlib import Path

# =============================================================================
# Project root
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# =============================================================================
# Data directories
# =============================================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

# =============================================================================
# Output directories
# =============================================================================

OUTPUT_DIR = ROOT_DIR / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

# =============================================================================
# Model directories
# =============================================================================

MODELS_DIR = ROOT_DIR / "models"

MODEL_V1_DIR = MODELS_DIR / "v1"

# =============================================================================
# Standard files
# =============================================================================

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "analytical_dataset.csv"
)

FINAL_DATA_FILE = (
    FINAL_DATA_DIR /
    "final_ml_dataset.csv"
)

MODEL_FILE = (
    MODEL_V1_DIR /
    "knn_model.joblib"
)

SCALER_FILE = (
    MODEL_V1_DIR /
    "standard_scaler.joblib"
)

METRICS_FILE = (
    MODEL_V1_DIR /
    "metrics.json"
)

# =============================================================================
# WHO indicator names
# =============================================================================

WHO_COLUMN_MAPPING = {

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    "Life expectancy (years) ":
        "life_expectancy",

    # -------------------------------------------------------------------------
    # Education
    # -------------------------------------------------------------------------

    "Expected years of schooling (children aged 6)":
        "expected_years_schooling",

    "Mean years of schooling (population aged 25+)":
        "mean_years_schooling",

    # -------------------------------------------------------------------------
    # Poverty
    # -------------------------------------------------------------------------

    "Multidimensional Poverty Index":
        "mpi",

    # -------------------------------------------------------------------------
    # Environmental health
    # -------------------------------------------------------------------------

    "Concentrations of fine particulate matter (PM2.5)":
        "pm25",

    "Population with primary reliance on polluting fuels and technologies for cooking (%)":
        "polluting_cooking_fuels",

    # -------------------------------------------------------------------------
    # Water, sanitation and hygiene
    # -------------------------------------------------------------------------

    "Population using safely managed drinking water services (%)":
        "safe_drinking_water",

    "Population using safely managed sanitation services (%)":
        "safe_sanitation",

    # -------------------------------------------------------------------------
    # Alcohol
    # -------------------------------------------------------------------------

    "Alcohol, consumers in past 12 months (age-standardized) (%)":
        "alcohol_consumers",

    "Alcohol, per capita consumption (15+ years, among drinkers only) (in litres of pure alcohol)":
        "alcohol_consumption",

    # -------------------------------------------------------------------------
    # Tobacco
    # -------------------------------------------------------------------------

    "Tobacco, current tobacco use prevalence (model-based estimates, age-standardized) (%)":
        "tobacco_use",

    # -------------------------------------------------------------------------
    # Noncommunicable diseases
    # -------------------------------------------------------------------------

    "Obesity prevalence among adults, BMI>=30 (age-standardized) (%)":
        "obesity_prevalence",

    # -------------------------------------------------------------------------
    # Health care
    # -------------------------------------------------------------------------

    "Medical doctors (%)":
        "medical_doctors",

    "Diabetes treatment coverage (30+ years) (age-standardized) (%)":
        "diabetes_treatment",

    "Hypertension treatment coverage among adults aged 30-79 with hypertension (age-standardized) (%)":
        "hypertension_treatment",
}

# =============================================================================
# Project directories
# =============================================================================

PROJECT_DIRECTORIES = (

    RAW_DATA_DIR,
    CACHE_DIR,

    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,

    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,

    MODELS_DIR,
    MODEL_V1_DIR,

)

# =============================================================================
# Create project directories
# =============================================================================

for directory in PROJECT_DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )