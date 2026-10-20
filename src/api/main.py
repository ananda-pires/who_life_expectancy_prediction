"""FastAPI application for WHO HIDR life expectancy predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import router
from src.api.services import load_model_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the application starts."""
    model, scaler = load_model_artifacts()
    app.state.model = model
    app.state.scaler = scaler

    yield

    app.state.model = None
    app.state.scaler = None


app = FastAPI(
    title="WHO HIDR Life Expectancy Prediction API",
    description=(
        "Predict population-level life expectancy from seven "
        "WHO health inequality indicators."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
