"""Red-light violation detector."""

from __future__ import annotations

from dataclasses import dataclass

from visiondrive.constants import EVENT_RED_LIGHT, SIGNAL_RED


@dataclass
class TrackState:
    last_y: float
    lane_id: str


class RedLightViolationDetector:
    """Flags a violation when a vehicle center crosses stop-line while its lane is RED."""

    def __init__(self, stop_line_y: int) -> None:
        self.stop_line_y = stop_line_y
        self._prev: dict[int, TrackState] = {}
        self._already_flagged: set[int] = set()

    def update(
        self,
        object_id: int,
        center_y: float,
        lane_id: str,
        signal_map: dict[str, str],
        frame_index: int,
    ) -> dict | None:
        previous = self._prev.get(object_id)
        self._prev[object_id] = TrackState(last_y=center_y, lane_id=lane_id)
        if previous is None or object_id in self._already_flagged:
            return None

        crossed = previous.last_y <= self.stop_line_y < center_y
        is_red = signal_map.get(lane_id, SIGNAL_RED) == SIGNAL_RED
        if crossed and is_red:
            self._already_flagged.add(object_id)
            return {
                "frame": frame_index,
                "object_id": object_id,
                "lane": lane_id,
                "event": EVENT_RED_LIGHT,
            }
        return None
