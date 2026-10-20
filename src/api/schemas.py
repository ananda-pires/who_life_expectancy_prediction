"""Pydantic schemas for the life expectancy prediction API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
SchoolingYears = Annotated[float, Field(ge=0, le=30, allow_inf_nan=False)]
AlcoholConsumption = Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """Validated input features required by the trained KNN model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    obesity_prevalence: Percentage
    diabetes_treatment: Percentage
    polluting_cooking_fuels: Percentage
    hypertension_treatment: Percentage
    mean_years_schooling: SchoolingYears
    expected_years_schooling: SchoolingYears
    alcohol_consumption: AlcoholConsumption


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    model_config = ConfigDict(extra="forbid")

    prediction: float = Field(
        description="Predicted life expectancy in years.",
        allow_inf_nan=False,
    )
    target: str = Field(default="life_expectancy")
    model_version: str = Field(default="v1")


class HealthResponse(BaseModel):
    """API health and model-readiness status."""

    status: str
    model_loaded: bool
    model_version: str
