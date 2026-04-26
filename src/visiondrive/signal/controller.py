"""
Adaptive signal controller. Pure logic - no threading, no I/O. Given lane
counts and tunables, returns per-lane green/yellow/red durations and the
priority lane.
"""

from __future__ import annotations

from visiondrive.settings import SignalConfig


class AdaptiveSignalController:
    def __init__(self, config: SignalConfig, saturation_per_lane: int) -> None:
        self.config = config
        self.saturation_per_lane = saturation_per_lane

    def _normalize(self, lane_counts: dict[str, int]) -> dict[str, float]:
        sat = max(1, self.saturation_per_lane)
        return {lane: min(max(0, count) / sat, 1.0) for lane, count in lane_counts.items()}

    def _green_budget(self, num_lanes: int) -> float:
        cfg = self.config
        raw = cfg.base_cycle_time - (num_lanes * cfg.yellow_time) - cfg.cycle_buffer
        return max(num_lanes * cfg.min_green, min(raw, num_lanes * cfg.max_green))

    def compute_timings(self, lane_counts: dict[str, int]) -> dict[str, dict[str, float]]:
        if not lane_counts:
            return {}

        lanes = list(lane_counts.keys())
        normalized = self._normalize(lane_counts)
        total_pressure = sum(normalized.values())
        if total_pressure <= 1e-9:
            shares = {lane: 1.0 / len(lanes) for lane in lanes}
        else:
            shares = {lane: normalized[lane] / total_pressure for lane in lanes}

        budget = self._green_budget(len(lanes))
        cfg = self.config
        greens = {lane: shares[lane] * budget for lane in lanes}
        for lane in lanes:
            greens[lane] += cfg.max_extension * (normalized[lane] ** 2)
            greens[lane] = max(cfg.min_green, min(greens[lane], cfg.max_green))

        timings: dict[str, dict[str, float]] = {}
        for lane in lanes:
            red = cfg.cycle_buffer
            for other in lanes:
                if other != lane:
                    red += greens[other] + cfg.yellow_time
            red = max(red, cfg.min_red)
            timings[lane] = {
                "green": round(greens[lane], 2),
                "yellow": round(cfg.yellow_time, 2),
                "red": round(red, 2),
            }
        return timings

    def pick_priority_lane(self, lane_counts: dict[str, int]) -> tuple[str, float]:
        if not lane_counts:
            raise ValueError("lane_counts cannot be empty")
        normalized = self._normalize(lane_counts)
        score = {lane: val + 0.5 * (val**2) for lane, val in normalized.items()}
        lane = max(score, key=score.get)
        timings = self.compute_timings(lane_counts)
        return lane, timings[lane]["green"]
