"""Loopback session and CSRF protection."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from summarizer_web.db.connection import get_database

SESSION_COOKIE = "summarizer_session"
CSRF_HEADER = "X-CSRF-Token"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
ALLOWED_ORIGINS = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    csrf_token: str


def _load_or_create_session(request: Request) -> tuple[SessionContext, bool]:
    session_id = request.cookies.get(SESSION_COOKIE)
    db = get_database()
    if session_id:
        row = db.fetchone(
            "SELECT session_id, csrf_token FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        if row is not None:
            return (
                SessionContext(session_id=row["session_id"], csrf_token=row["csrf_token"]),
                False,
            )
    session_id = str(uuid.uuid4())
    csrf_token = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO sessions (session_id, csrf_token, created_at) VALUES (?, ?, ?)",
        (session_id, csrf_token, datetime.now(timezone.utc).isoformat()),
    )
    return SessionContext(session_id=session_id, csrf_token=csrf_token), True


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        host = request.headers.get("host", "").split(":")[0]
        if host not in ALLOWED_HOSTS:
            return Response("Invalid host", status_code=400)

        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return Response("Invalid origin", status_code=403)

        session, created = _load_or_create_session(request)

        if request.method in {"POST", "PATCH", "PUT", "DELETE"} and request.url.path.startswith("/api/"):
            token = request.headers.get(CSRF_HEADER)
            if token != session.csrf_token:
                return Response("Invalid CSRF token", status_code=403)

        response = await call_next(request)

        if request.url.path.startswith("/api/"):
            if created:
                response.set_cookie(
                    SESSION_COOKIE,
                    session.session_id,
                    httponly=True,
                    samesite="strict",
                    secure=False,
                )
            response.headers["X-CSRF-Token"] = session.csrf_token

        return response


def get_session(request: Request) -> SessionContext:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=403, detail="Missing session")
    row = get_database().fetchone(
        "SELECT session_id, csrf_token FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    if row is None:
        raise HTTPException(status_code=403, detail="Invalid session")
    return SessionContext(session_id=row["session_id"], csrf_token=row["csrf_token"])
