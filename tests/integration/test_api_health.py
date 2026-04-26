"""
Smoke test for the API surface. We bypass the lifespan (which starts the
real pipeline + opens the video) by constructing a stripped-down app and
attaching a DataBus directly.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from visiondrive.api.routes import router
from visiondrive.core.data_bus import DataBus
from visiondrive.settings import Settings


def _make_app() -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings()
    app.state.data_bus = DataBus()
    app.include_router(router)
    return app


def test_health_returns_version():
    client = TestClient(_make_app())
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_state_returns_databus_snapshot():
    app = _make_app()
    app.state.data_bus.update(vehicle_count=5, fps=12.3)
    res = TestClient(app).get("/api/state")
    assert res.status_code == 200
    body = res.json()
    assert body["vehicle_count"] == 5
    assert body["fps"] == 12.3
