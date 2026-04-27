"""
Layered configuration loader.

Resolution order (later wins):
  1. config/config.yaml (committed defaults)
  2. config/config.{env}.yaml where env = VISIONDRIVE_ENV (dev/prod)
  3. .env / process environment, prefixed with VISIONDRIVE_ and nested via __

Example:
  VISIONDRIVE_API__PORT=8080 -> settings.api.port == 8080
  VISIONDRIVE_FEATURES__ENABLE_SMS=true -> settings.features.enable_sms is True

`get_settings()` is cached; tests can call `reload_settings()` to reset.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5000
    recompute_interval_sec: float = 2.0
    state_poll_interval_ms: int = 1000


class VideoConfig(BaseModel):
    path: str = "data/videos/traffic.mp4"
    output_path: str = "data/videos/hackathon_demo_output.avi"
    output_fps: float = 25.0


class ModelConfig(BaseModel):
    framework: str = "ultralytics"
    weights_path: str = "data/weights/yolov8n.pt"
    tracker_config: str = "bytetrack.yaml"
    conf: float = 0.15
    iou: float = 0.45
    imgsz: int = 960


class DetectionConfig(BaseModel):
    vehicle_class_ids: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 7])
    saturation_per_lane: int = 20
    stop_line_ratio_y: float = 0.62


class LanesConfig(BaseModel):
    ids: list[str] = Field(default_factory=lambda: ["lane_1", "lane_2", "lane_3", "lane_4"])


class SignalConfig(BaseModel):
    min_green: float = 8.0
    max_green: float = 45.0
    min_red: float = 8.0
    yellow_time: float = 3.0
    cycle_buffer: float = 2.0
    base_cycle_time: float = 90.0
    max_extension: float = 15.0


class FeaturesConfig(BaseModel):
    display_window: bool = True
    enable_helmet: bool = False
    enable_accident: bool = False
    enable_sms: bool = False


class HelmetConfig(BaseModel):
    weights_path: str = "data/weights/helmet_detector.pt"
    plate_weights_path: str | None = None
    bike_class_id: int = 3
    bike_conf: float = 0.20
    nested_conf: float = 0.35
    no_helmet_min_conf: float = 0.45
    fines_cooldown_sec: float = 5.0


class AccidentConfig(BaseModel):
    fps: float = 25.0
    overlap_iou_threshold: float = 0.35
    stationary_seconds: float = 2.0
    stationary_pixel_tolerance: float = 18.0
    high_speed_px_per_frame: float = 16.0
    near_zero_speed_px_per_frame: float = 1.8
    speed_drop_window_frames: int = 2
    incidents_dir: str = "incidents"
    camera_id: str = "CAM-01"
    location_name: str = "Main Junction - Smart City Demo"
    gps_lat: float | None = None
    gps_lon: float | None = None


class NotificationsConfig(BaseModel):
    cooldown_seconds: float = 60.0
    log_csv_path: str = "emergency_logs.csv"
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_phone: str | None = None
    emergency_phone: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must define a mapping at the top level")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Settings(BaseSettings):
    """Top-level settings tree. Values flow yaml -> env -> overrides."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONDRIVE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    log_level: str = "INFO"

    api: APIConfig = Field(default_factory=APIConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    lanes: LanesConfig = Field(default_factory=LanesConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    helmet: HelmetConfig = Field(default_factory=HelmetConfig)
    accident: AccidentConfig = Field(default_factory=AccidentConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)

    @property
    def project_root(self) -> Path:
        return _project_root()

    def resolve_path(self, value: str) -> Path:
        """Resolve a path from config relative to the project root."""
        p = Path(value)
        return p if p.is_absolute() else (self.project_root / p)


def _build_settings() -> Settings:
    root = _project_root()
    base_path = root / "config" / "config.yaml"
    base = _load_yaml(base_path)

    env_name = os.getenv("VISIONDRIVE_ENV", base.get("env", "dev"))
    overlay_path = root / "config" / f"config.{env_name}.yaml"
    merged = _deep_merge(base, _load_yaml(overlay_path))

    # Pydantic-settings will additionally apply VISIONDRIVE_* env overrides on top.
    return Settings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _build_settings()


def reload_settings() -> Settings:
    """Force-rebuild the cached settings (use in tests after env mutation)."""
    get_settings.cache_clear()
    return get_settings()
