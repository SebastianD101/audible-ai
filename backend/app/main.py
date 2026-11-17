"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.api import projections, recommendations, sleeper
from backend.app.config import get_settings
from backend.app.projections import NoProjectionsError

app = FastAPI(title="Audible AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(projections.router)
app.include_router(recommendations.router)
app.include_router(sleeper.router)


@app.exception_handler(NoProjectionsError)
@app.exception_handler(FileNotFoundError)
async def data_not_ready(_: Request, exc: Exception) -> JSONResponse:
    """Missing projections or rosters: the service is up but has no data to serve yet."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str


@app.get("/health")
def health() -> HealthResponse:
    """Report that the service is up."""
    return HealthResponse(status="ok")


def mount_frontend(target: FastAPI, directory: Path | None) -> None:
    """Serve the static frontend at ``/``, if one is configured.

    Call it after every API route is registered: Starlette matches routes in order, so
    ``/api/*`` and ``/health`` win and everything else falls through to the UI files.
    """
    if directory is None:
        return
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"STATIC_DIR {directory} has no index.html; build the frontend")
    target.mount("/", StaticFiles(directory=directory, html=True), name="frontend")


mount_frontend(app, get_settings().static_dir)
