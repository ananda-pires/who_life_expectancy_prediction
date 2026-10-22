"""Pydantic schemas for the life expectancy prediction API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
SchoolingYears = Annotated[float, Field(ge=0, le=30, allow_inf_nan=False)]
AlcoholConsumption = Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """Validated input features required by the trained prediction pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True)

    obesity_prevalence: Percentage = Field(
        description="Adult obesity prevalence, expressed as a percentage (0–100)."
    )
    diabetes_treatment: Percentage = Field(
        description="Diabetes treatment coverage, expressed as a percentage (0–100)."
    )
    polluting_cooking_fuels: Percentage = Field(
        description="Prevalence of polluting cooking fuels, expressed as a percentage (0–100)."
    )
    hypertension_treatment: Percentage = Field(
        description="Hypertension treatment coverage, expressed as a percentage (0–100)."
    )
    mean_years_schooling: SchoolingYears = Field(
        description="Mean years of schooling (0–30 years)."
    )
    expected_years_schooling: SchoolingYears = Field(
        description="Expected years of schooling (0–30 years)."
    )
    alcohol_consumption: AlcoholConsumption = Field(
        description="Alcohol consumption indicator on the dataset scale (0–50)."
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    model_config = ConfigDict(extra="forbid")

    prediction: float = Field(
        description="Predicted life expectancy in years.",
        allow_inf_nan=False,
    )
    target: str = Field(default="life_expectancy")
    model_version: str = Field(default="v2")


class HealthResponse(BaseModel):
    """API health and model-readiness status."""

    status: str
    model_loaded: bool
    model_version: str
