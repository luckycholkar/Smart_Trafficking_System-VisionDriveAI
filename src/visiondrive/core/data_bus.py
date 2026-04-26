"""
Thread-safe shared state.

The DataBus is the contract between the vision pipeline (producer of vehicle
counts, FPS, violations), the signal runner (producer of signal_state), and
the API (consumer for /api/state). Always go through here - never call across
threads directly.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

from visiondrive.constants import (
    BUS_FPS,
    BUS_FRAME_INDEX,
    BUS_LANE_COUNTS,
    BUS_SIGNAL_STATE,
    BUS_VEHICLE_COUNT,
    BUS_VIOLATIONS,
    DEFAULT_VIOLATIONS_KEEP,
)


class DataBus:
    def __init__(self, *, violations_keep: int = DEFAULT_VIOLATIONS_KEEP) -> None:
        self._lock = threading.Lock()
        self._violations_keep = violations_keep
        self._state: dict[str, Any] = {
            BUS_FRAME_INDEX: 0,
            BUS_VEHICLE_COUNT: 0,
            BUS_LANE_COUNTS: {},
            BUS_SIGNAL_STATE: {},
            BUS_VIOLATIONS: [],
            BUS_FPS: 0.0,
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._state.update(kwargs)

    def append_violation(self, violation: dict[str, Any]) -> None:
        with self._lock:
            violations = self._state.setdefault(BUS_VIOLATIONS, [])
            violations.append(violation)
            self._state[BUS_VIOLATIONS] = violations[-self._violations_keep :]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)
