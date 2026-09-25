"""Typed API errors and the handlers that render them as ErrorResponse JSON."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An error the client can act on, identified by a stable `code`."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.retryable = retryable


def error_body(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details, "retryable": retryable}


def _validation_message(error: RequestValidationError) -> str:
    parts = []
    for item in error.errors()[:5]:
        location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
        parts.append(f"{location}: {item.get('msg', 'invalid')}" if location else item.get("msg", "invalid"))
    return "; ".join(parts) or "Invalid request"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            error_body(
                error.code, error.message, details=error.details, retryable=error.retryable
            ),
            status_code=error.status,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body(
                "invalid_request",
                _validation_message(error),
                details={"errors": jsonable_encoder(error.errors())},
            ),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        message = error.detail if isinstance(error.detail, str) else "Request failed"
        return JSONResponse(
            error_body(f"http_{error.status_code}", message),
            status_code=error.status_code,
            headers=getattr(error, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=error)
        return JSONResponse(
            error_body("internal_error", "Unexpected server error. Check the server log."),
            status_code=500,
        )
