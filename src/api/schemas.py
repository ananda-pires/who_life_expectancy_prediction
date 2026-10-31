"""Pydantic schemas for the life expectancy prediction API.

Ranges are operational plausibility limits chosen from the observed data
(2000-2022) with some margin. They are NOT official WHO thresholds.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# field -> (minimum, maximum, unit), also used to build friendly error messages
RANGES = {
    "obesity_prevalence": (0, 100, "%"),
    "diabetes_treatment": (0, 100, "%"),
    "polluting_cooking_fuels": (0, 100, "%"),
    "hypertension_treatment": (0, 100, "%"),
    "mean_years_schooling": (0, 20, "years"),
    "expected_years_schooling": (0, 25, "years"),
    "alcohol_consumption": (0, 100, "litres of pure alcohol"),
}

Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
MeanSchooling = Annotated[float, Field(ge=0, le=20, allow_inf_nan=False)]
ExpectedSchooling = Annotated[float, Field(ge=0, le=25, allow_inf_nan=False)]
AlcoholLitres = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """Seven predictors required by the trained pipeline (all required)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    obesity_prevalence: Percentage = Field(
        description="Obesity prevalence among adults, BMI>=30, age-standardized (%). Range 0-100."
    )
    diabetes_treatment: Percentage = Field(
        description="Diabetes treatment coverage, 30+ years, age-standardized (%). Range 0-100."
    )
    polluting_cooking_fuels: Percentage = Field(
        description="Population relying primarily on polluting fuels for cooking (%). Range 0-100."
    )
    hypertension_treatment: Percentage = Field(
        description="Hypertension treatment coverage, adults 30-79 with hypertension (%). Range 0-100."
    )
    mean_years_schooling: MeanSchooling = Field(
        description="Mean years of schooling, population aged 25+ (years). Range 0-20."
    )
    expected_years_schooling: ExpectedSchooling = Field(
        description="Expected years of schooling, children aged 6 (years). Range 0-25."
    )
    alcohol_consumption: AlcoholLitres = Field(
        description=(
            "Per capita alcohol consumption, 15+ years, among drinkers only "
            "(litres of pure alcohol). Range 0-100."
        )
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    model_config = ConfigDict(extra="forbid")

    prediction: float = Field(
        description="Predicted life expectancy in years.", allow_inf_nan=False
    )
    target: str = Field(default="life_expectancy")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """API health and model-readiness status."""

    status: str
    model_loaded: bool
    model_name: str | None
    model_version: str
