"""
Detector abstraction layer.

The pipeline talks to detectors only through `Detector` (see base.py). New
model frameworks plug in by registering an implementation under a name in
`registry`. See docs/models.md for a walkthrough.
"""

from visiondrive.models.base import Detection, Detector, TrackedDetection
from visiondrive.models.factory import build_detector
from visiondrive.models.registry import detector_registry, register_detector

# Importing the implementation modules registers them as a side-effect.
from visiondrive.models import ultralytics  # noqa: F401  (registration)

__all__ = [
    "Detection",
    "Detector",
    "TrackedDetection",
    "build_detector",
    "detector_registry",
    "register_detector",
]
