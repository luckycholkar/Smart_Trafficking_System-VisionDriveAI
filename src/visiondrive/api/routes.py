"""API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from visiondrive import __version__
from visiondrive.api.schemas import HealthResponse, StateResponse
from visiondrive.core.data_bus import DataBus

router = APIRouter()
_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _data_bus(request: Request) -> DataBus:
    return request.app.state.data_bus


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@router.get("/api/state", response_model=StateResponse)
def state(request: Request) -> StateResponse:
    return StateResponse(**_data_bus(request).snapshot())


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    return _templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "version": __version__,
            "poll_interval_ms": settings.api.state_poll_interval_ms,
        },
    )
