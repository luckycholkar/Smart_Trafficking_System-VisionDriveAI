from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SignalConfig:
    min_green: float = 8.0
    max_green: float = 45.0
    min_red: float = 8.0
    yellow_time: float = 3.0
    cycle_buffer: float = 2.0
    saturation_per_lane: int = 20
    base_cycle_time: float = 90.0
    max_extension: float = 15.0


class AdaptiveSignalController:
    """
    Converts lane counts into dynamic timings for a software-only signal demo.
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def _normalize(self, lane_counts: Dict[str, int]) -> Dict[str, float]:
        sat = max(1, self.config.saturation_per_lane)
        return {lane: min(max(0, count) / sat, 1.0) for lane, count in lane_counts.items()}

    def _green_budget(self, lane_count: int) -> float:
        cfg = self.config
        raw = cfg.base_cycle_time - (lane_count * cfg.yellow_time) - cfg.cycle_buffer
        return max(lane_count * cfg.min_green, min(raw, lane_count * cfg.max_green))

    def compute_timings(self, lane_counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        if not lane_counts:
            return {}

        lanes: List[str] = list(lane_counts.keys())
        normalized = self._normalize(lane_counts)
        total_pressure = sum(normalized.values())
        if total_pressure <= 1e-9:
            shares = {lane: 1.0 / len(lanes) for lane in lanes}
        else:
            shares = {lane: normalized[lane] / total_pressure for lane in lanes}

        budget = self._green_budget(len(lanes))
        greens = {lane: shares[lane] * budget for lane in lanes}
        for lane in lanes:
            greens[lane] += self.config.max_extension * (normalized[lane] ** 2)
            greens[lane] = max(self.config.min_green, min(greens[lane], self.config.max_green))

        timings: Dict[str, Dict[str, float]] = {}
        for lane in lanes:
            red = self.config.cycle_buffer
            for other in lanes:
                if other != lane:
                    red += greens[other] + self.config.yellow_time
            red = max(red, self.config.min_red)
            timings[lane] = {
                "green": round(greens[lane], 2),
                "yellow": round(self.config.yellow_time, 2),
                "red": round(red, 2),
            }
        return timings

    def pick_priority_lane(self, lane_counts: Dict[str, int]) -> Tuple[str, float]:
        if not lane_counts:
            raise ValueError("lane_counts cannot be empty")
        normalized = self._normalize(lane_counts)
        score = {lane: val + 0.5 * (val**2) for lane, val in normalized.items()}
        lane = max(score, key=score.get)
        timings = self.compute_timings(lane_counts)
        return lane, timings[lane]["green"]

