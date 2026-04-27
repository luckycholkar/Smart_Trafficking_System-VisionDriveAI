"""
FastAPI factory.

The app's lifespan hook spins up:
  - the vision pipeline (detector + reader/inference threads),
  - the signal runner thread,

and tears them down on shutdown. The DataBus is shared across components and
exposed via app.state.data_bus so request handlers can snapshot it.

Run locally:
    uvicorn visiondrive.api.app:app --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from visiondrive import __version__
from visiondrive.api.routes import router
from visiondrive.core.data_bus import DataBus
from visiondrive.core.pipeline import VisionPipeline
from visiondrive.models import build_detector
from visiondrive.settings import Settings, get_settings
from visiondrive.signal.runner import SignalRunner

log = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    _configure_logging(settings)

    data_bus = DataBus()
    app.state.data_bus = data_bus

    detector = build_detector(settings.model)
    pipeline = VisionPipeline(settings, detector, data_bus)
    signal_runner = SignalRunner(settings, data_bus)

    pipeline_thread = threading.Thread(target=pipeline.run, name="vd-pipeline", daemon=True)
    signal_thread = threading.Thread(target=signal_runner.run, name="vd-signal", daemon=True)

    pipeline_thread.start()
    signal_thread.start()
    log.info("VisionDrive started (model=%s, video=%s)", settings.model.framework, settings.video.path)

    try:
        yield
    finally:
        pipeline.stop_event.set()
        signal_runner.stop_event.set()
        pipeline_thread.join(timeout=5.0)
        signal_thread.join(timeout=2.0)
        log.info("VisionDrive stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="VisionDrive AI",
        version=__version__,
        description="Smart City traffic monitoring + adaptive signal control.",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    return app


app = create_app()
