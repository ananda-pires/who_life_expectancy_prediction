"""HTTP routes for the life expectancy prediction API."""

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services import MODEL_VERSION, predict_life_expectancy


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Report whether the API and its model are ready."""
    model_loaded = (
        getattr(request.app.state, "model", None) is not None
        and getattr(request.app.state, "scaler", None) is not None
    )

    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Predict life expectancy from validated health indicators."""
    model = getattr(request.app.state, "model", None)
    scaler = getattr(request.app.state, "scaler", None)

    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is not available.",
        )

    try:
        prediction = predict_life_expectancy(
            model=model,
            scaler=scaler,
            features=payload.model_dump(),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc

    return PredictionResponse(prediction=prediction)
