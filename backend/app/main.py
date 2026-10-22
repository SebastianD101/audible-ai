"""FastAPI application entry point."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Audible AI", version="0.1.0")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str


@app.get("/health")
def health() -> HealthResponse:
    """Report that the service is up."""
    return HealthResponse(status="ok")
