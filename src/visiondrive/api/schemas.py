"""Pydantic response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class StateResponse(BaseModel):
    """Live snapshot of the DataBus."""

    frame_index: int = 0
    vehicle_count: int = 0
    fps: float = 0.0
    congestion_score: float = 0.0
    active_green_lane: str | None = None
    green_remaining: float = 0.0
    lane_counts: dict[str, int] = Field(default_factory=dict)
    signal_state: dict[str, str] = Field(default_factory=dict)
    signal_timings: dict[str, dict[str, float]] = Field(default_factory=dict)
    violations: list[dict[str, Any]] = Field(default_factory=list)
