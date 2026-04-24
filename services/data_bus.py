from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any, Dict


class DataBus:
    """
    Thread-safe in-memory store for live demo telemetry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "frame_index": 0,
            "vehicle_count": 0,
            "lane_counts": {},
            "signal_state": {},
            "violations": [],
            "fps": 0.0,
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._state.update(kwargs)

    def append_violation(self, violation: Dict[str, Any]) -> None:
        with self._lock:
            violations = self._state.setdefault("violations", [])
            violations.append(violation)
            self._state["violations"] = violations[-30:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

