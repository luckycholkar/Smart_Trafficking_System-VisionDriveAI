from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass
class TrackState:
    last_y: float
    lane_id: str


class RedLightViolationDetector:
    """
    Flags a violation when a vehicle center crosses stop-line while its lane is RED.
    """

    def __init__(self, stop_line_y: int) -> None:
        self.stop_line_y = stop_line_y
        self.prev: Dict[int, TrackState] = {}
        self.already_flagged: Set[int] = set()

    def update(
        self,
        object_id: int,
        center_y: float,
        lane_id: str,
        signal_map: Dict[str, str],
        frame_index: int,
    ) -> Optional[dict]:
        previous = self.prev.get(object_id)
        self.prev[object_id] = TrackState(last_y=center_y, lane_id=lane_id)
        if previous is None or object_id in self.already_flagged:
            return None

        crossed = previous.last_y <= self.stop_line_y < center_y
        is_red = signal_map.get(lane_id, "RED") == "RED"
        if crossed and is_red:
            self.already_flagged.add(object_id)
            return {
                "frame": frame_index,
                "object_id": object_id,
                "lane": lane_id,
                "event": "STOP_LINE_CROSSED_ON_RED",
            }
        return None

