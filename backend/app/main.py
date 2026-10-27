"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.api import projections
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
