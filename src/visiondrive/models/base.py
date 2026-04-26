"""
Detector contract. All model frameworks (Ultralytics today, others tomorrow)
implement this interface so the pipeline stays framework-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Detection:
    """A single detection in image coordinates."""

    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    class_id: int
    confidence: float


@dataclass(frozen=True)
class TrackedDetection(Detection):
    """A detection with a stable tracker-assigned identity."""

    track_id: int


class Detector(ABC):
    """Framework-agnostic vehicle detector interface."""

    @abstractmethod
    def detect(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[Detection]:
        """Run a single forward pass and return detections."""

    @abstractmethod
    def track(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[TrackedDetection]:
        """Run detection + tracking. Returns detections with persistent track IDs."""

    @property
    @abstractmethod
    def framework(self) -> str:
        """Short framework identifier (e.g. 'ultralytics')."""
