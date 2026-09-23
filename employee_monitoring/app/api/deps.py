"""Shared API dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is not ready")
    return service


def require_api_key(request: Request) -> None:
    service = get_service(request)
    expected = service.config.api_key
    if not expected:
        if service.config.allow_insecure_api:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured",
        )
    provided = request.headers.get("X-API-Key", "")
    if provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
