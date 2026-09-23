"""HTTP API. The vision pipeline starts beside it and does not live inside a request."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import attendance, cameras, employees, monitoring
from app.config import load_config
from app.database.session import session_scope
from app.logging_config import configure_logging
from app.service import PipelineService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service: PipelineService = app.state.service
    service.start()
    if service.config.evidence.enabled:
        service.evidence.purge_expired()
    yield
    service.stop()


def create_app(service: PipelineService | None = None) -> FastAPI:
    config = service.config if service is not None else load_config()
    configure_logging(config.log_level)
    app = FastAPI(title="Employee Monitoring", lifespan=lifespan)
    app.state.service = service or PipelineService(config)
    app.include_router(employees.router)
    app.include_router(attendance.router)
    app.include_router(monitoring.router)
    app.include_router(cameras.router)

    @app.middleware("http")
    async def audit_access(request, call_next):
        response = await call_next(request)
        current = getattr(request.app.state, "service", None)
        factory = getattr(current, "_session_factory", None) if current else None
        if factory is not None and request.url.path != "/health":
            try:
                with session_scope(factory) as db:
                    current.repository.audit(db, request.method, request.url.path, response.status_code)
            except Exception:
                logger.warning("audit write skipped", extra={"event": "AUDIT_SKIPPED"})
        return response

    @app.get("/health")
    def process_health() -> dict:
        current: PipelineService = app.state.service
        return {"status": "ok", "database": current._db_ready, "pipeline": current.config.run_pipeline}

    return app


app = create_app()
