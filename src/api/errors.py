"""Consistent, safe error responses for the prediction API."""

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import RANGES

logger = logging.getLogger("life_expectancy_api")

_RULES = {
    "missing": ("required field is missing", "Include this field in the JSON body."),
    "extra_forbidden": (
        "field is not accepted",
        "Remove this field; only the seven documented predictors are allowed.",
    ),
    "float_type": ("a JSON number is required", "Send a number (for example 25.5), not text."),
    "float_parsing": ("a JSON number is required", "Send a number (for example 25.5), not text."),
    "greater_than_equal": ("value is below the allowed minimum", "Use a value inside the expected range."),
    "less_than_equal": ("value is above the allowed maximum", "Use a value inside the expected range."),
    "finite_number": ("value must be finite", "Do not send NaN or infinity."),
}


def _error_body(code, message, details=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


def _friendly(err):
    loc = [str(p) for p in err.get("loc", []) if p != "body"]
    field = ".".join(loc) or "body"
    rule, hint = _RULES.get(
        err.get("type", ""),
        ("value is invalid", "Check the field against the API documentation."),
    )
    item = {"field": field, "rule": rule, "how_to_fix": hint}
    if field in RANGES:
        low, high, unit = RANGES[field]
        item["expected"] = f"number between {low} and {high} ({unit})"
    return item


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errors):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "malformed_json",
                    "The request body is not valid JSON.",
                    [{"how_to_fix": "Send a syntactically valid JSON object."}],
                ),
            )
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                _error_body(
                    "validation_error",
                    "The request body does not match the expected schema.",
                    [_friendly(e) for e in errors],
                )
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.exception("Unexpected error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "An unexpected error occurred."),
        )
