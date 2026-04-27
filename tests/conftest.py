"""
Pytest fixtures.

We mock the Detector boundary so unit tests can run without ultralytics or
weights on disk - that's the whole point of having an abstraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visiondrive.models.base import Detection, Detector, TrackedDetection  # noqa: E402
from visiondrive.settings import Settings  # noqa: E402


class FakeDetector(Detector):
    """Returns a scripted list of detections per call. Useful for pipeline tests."""

    framework_name = "fake"

    def __init__(self, scripted: list[list[TrackedDetection]] | None = None) -> None:
        self._scripted = scripted or []
        self._idx = 0

    @property
    def framework(self) -> str:
        return self.framework_name

    def detect(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[Detection]:
        return []

    def track(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[TrackedDetection]:
        if self._idx >= len(self._scripted):
            return []
        out = self._scripted[self._idx]
        self._idx += 1
        return out


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
