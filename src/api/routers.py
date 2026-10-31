"""HTTP routes for the life expectancy prediction API."""

import logging

from fastapi import APIRouter, HTTPException, Request, Response, status

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.api.services import MODEL_VERSION, predict_life_expectancy

logger = logging.getLogger("life_expectancy_api")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request, response: Response) -> HealthResponse:
    """200 when the model is loaded; 503 when the service cannot predict."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
        model_name=getattr(request.app.state, "model_name", None),
        model_version=MODEL_VERSION,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict life expectancy using the model loaded at startup."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is not available.",
        )
    try:
        prediction = predict_life_expectancy(model=model, features=payload.model_dump())
    except (ValueError, RuntimeError) as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc
    return PredictionResponse(
        prediction=prediction,
        model_name=getattr(request.app.state, "model_name", None) or "unknown",
        model_version=MODEL_VERSION,
    )
