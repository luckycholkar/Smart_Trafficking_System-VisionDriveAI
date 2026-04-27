"""Build a Detector from settings."""

from __future__ import annotations

from visiondrive.models.base import Detector
from visiondrive.models.registry import get_detector_factory
from visiondrive.settings import ModelConfig


def build_detector(model_cfg: ModelConfig, weights_path_override: str | None = None) -> Detector:
    """
    Construct a `Detector` for the configured framework.

    `weights_path_override` lets violation pipelines (helmet, plate) reuse the
    factory with their own weights while sharing detector params.
    """
    factory = get_detector_factory(model_cfg.framework)
    return factory(
        weights_path=weights_path_override or model_cfg.weights_path,
        conf=model_cfg.conf,
        iou=model_cfg.iou,
        imgsz=model_cfg.imgsz,
        tracker_config=model_cfg.tracker_config,
    )
