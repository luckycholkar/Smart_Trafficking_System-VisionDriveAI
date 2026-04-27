"""
Immutable constants. Anything here is a fact about the world or the data
schema (e.g. COCO class IDs) - not a tunable knob. Tunable values live in
config.yaml and are surfaced via visiondrive.settings.
"""

from __future__ import annotations

from typing import Final

# COCO class IDs used by Ultralytics YOLO checkpoints.
COCO_CLASS_NAMES: Final[dict[int, str]] = {
    1: "Cycle",
    2: "Car",
    3: "Bike/Scooter",
    5: "Bus",
    7: "Truck",
}

# Signal states.
SIGNAL_GREEN: Final[str] = "GREEN"
SIGNAL_YELLOW: Final[str] = "YELLOW"
SIGNAL_RED: Final[str] = "RED"

# Violation event types.
EVENT_RED_LIGHT: Final[str] = "STOP_LINE_CROSSED_ON_RED"
EVENT_HELMET_FINE: Final[str] = "NO_HELMET_FINE"
EVENT_ACCIDENT_IMPACT: Final[str] = "IMPACT_ALERT"
EVENT_ACCIDENT_COLLISION: Final[str] = "COLLISION_STATIONARY"

# DataBus keys (single source of truth so producers/consumers can't drift).
BUS_FRAME_INDEX: Final[str] = "frame_index"
BUS_VEHICLE_COUNT: Final[str] = "vehicle_count"
BUS_LANE_COUNTS: Final[str] = "lane_counts"
BUS_SIGNAL_STATE: Final[str] = "signal_state"
BUS_SIGNAL_TIMINGS: Final[str] = "signal_timings"
BUS_ACTIVE_GREEN_LANE: Final[str] = "active_green_lane"
BUS_GREEN_REMAINING: Final[str] = "green_remaining"
BUS_CONGESTION: Final[str] = "congestion_score"
BUS_VIOLATIONS: Final[str] = "violations"
BUS_FPS: Final[str] = "fps"

# Project paths relative to repo root.
DEFAULT_CONFIG_PATH: Final[str] = "config/config.yaml"
DEFAULT_VIOLATIONS_KEEP: Final[int] = 30
