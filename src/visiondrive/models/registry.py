"""
Detector registry. Add a new framework by calling `register_detector("name", Cls)`
in your implementation module - the import side-effect from
visiondrive.models.__init__ wires it up.
"""

from __future__ import annotations

from typing import Callable

from visiondrive.models.base import Detector

DetectorFactory = Callable[..., Detector]

detector_registry: dict[str, DetectorFactory] = {}


def register_detector(name: str, factory: DetectorFactory) -> None:
    if name in detector_registry:
        raise ValueError(f"Detector '{name}' is already registered")
    detector_registry[name] = factory


def get_detector_factory(name: str) -> DetectorFactory:
    if name not in detector_registry:
        available = ", ".join(sorted(detector_registry)) or "<none>"
        raise KeyError(f"Unknown detector framework '{name}'. Available: {available}")
    return detector_registry[name]
