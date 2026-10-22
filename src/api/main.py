"""FastAPI application for WHO HIDR life expectancy predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import router
from src.api.services import load_model_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained prediction pipeline when the application starts."""
    app.state.model = load_model_artifacts()

    yield

    app.state.model = None


app = FastAPI(
    title="WHO HIDR Life Expectancy Prediction API",
    description=(
        "Predict population-level life expectancy from seven "
        "WHO health inequality indicators."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)
