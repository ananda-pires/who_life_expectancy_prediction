"""FastAPI application for WHO HIDR life expectancy predictions."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.errors import register_exception_handlers
from src.api.routers import router
from src.api.services import load_model_artifacts, read_model_name

logger = logging.getLogger("life_expectancy_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once per process, before requests are accepted.

    If the artifact is missing or corrupted, the app starts in degraded mode:
    /health and /predict answer 503 and the error is logged.
    """
    try:
        app.state.model = load_model_artifacts()
        app.state.model_name = read_model_name()
        logger.info("Model loaded at startup: %s", app.state.model_name)
    except Exception:
        logger.exception("Model could not be loaded; service starts unavailable.")
        app.state.model = None
        app.state.model_name = None

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

register_exception_handlers(app)
app.include_router(router)
