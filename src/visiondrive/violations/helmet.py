"""
Helmet violation pipeline.

Two-stage nested detection:
  1. Detect two-wheelers from the primary detector.
  2. Run a nested helmet detector on each two-wheeler crop.
  3. On no-helmet, run ANPR on the plate ROI and issue a fine payload.

Uses the Detector abstraction so swapping frameworks (e.g. yolov8 -> yolov10
or a custom model) is a registration change, not a rewrite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import cv2

from visiondrive.models import Detector
from visiondrive.violations._anpr import run_anpr

BBox = tuple[int, int, int, int]


@dataclass
class HelmetViolation:
    bike_bbox: BBox
    plate_bbox: BBox | None
    bike_conf: float
    no_helmet_conf: float
    plate_number: str
    anpr_confidence: str
    fine_id: str
    timestamp: str


def _clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> BBox:
    return (
        max(0, min(int(x1), w - 1)),
        max(0, min(int(y1), h - 1)),
        max(0, min(int(x2), w - 1)),
        max(0, min(int(y2), h - 1)),
    )


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


class HelmetViolationPipeline:
    def __init__(
        self,
        vehicle_detector: Detector,
        helmet_detector: Detector,
        plate_detector: Detector | None = None,
        *,
        motorcycle_class_id: int = 3,
        rider_class_ids: tuple[int, ...] = (0,),
        no_helmet_class_ids: tuple[int, ...] = (2,),
        no_helmet_min_conf: float = 0.45,
        fines_cooldown_sec: float = 5.0,
        anpr_fn: Callable[[Any], dict[str, str]] = run_anpr,
    ) -> None:
        self._vehicle = vehicle_detector
        self._helmet = helmet_detector
        self._plate = plate_detector
        self._motorcycle_class_id = motorcycle_class_id
        self._rider_class_ids = set(rider_class_ids)
        self._no_helmet_class_ids = set(no_helmet_class_ids)
        self._no_helmet_min_conf = no_helmet_min_conf
        self._fines_cooldown_sec = fines_cooldown_sec
        self._anpr_fn = anpr_fn
        self._recent_fines: list[tuple[BBox, float]] = []

    def detect_violations(self, frame: Any) -> list[HelmetViolation]:
        h, w = frame.shape[:2]
        violations: list[HelmetViolation] = []
        bikes = self._vehicle.detect(frame, classes=[self._motorcycle_class_id])

        for bike in bikes:
            x1, y1, x2, y2 = _clip_bbox(*bike.bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            bike_crop = frame[y1:y2, x1:x2]
            if bike_crop.size == 0:
                continue

            nested = self._helmet.detect(bike_crop)
            if not self._is_no_helmet(nested):
                continue

            score = self._max_no_helmet_conf(nested)
            if score < self._no_helmet_min_conf:
                continue
            if self._is_duplicate_violation((x1, y1, x2, y2)):
                continue

            plate_bbox_global, plate_crop = self._extract_plate_roi(
                frame, bike_crop, (x1, y1, x2, y2)
            )
            plate_info = self._safe_anpr(plate_crop)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._recent_fines.append(((x1, y1, x2, y2), time.time()))
            violations.append(
                HelmetViolation(
                    bike_bbox=(x1, y1, x2, y2),
                    plate_bbox=plate_bbox_global,
                    bike_conf=bike.confidence,
                    no_helmet_conf=score,
                    plate_number=plate_info["plate_number"],
                    anpr_confidence=plate_info["confidence"],
                    fine_id=f"FINE-{datetime.now().strftime('%H%M%S%f')[-8:]}",
                    timestamp=ts,
                )
            )

        return violations

    def _is_no_helmet(self, nested: list) -> bool:
        if not nested:
            return False
        has_rider = any(d.class_id in self._rider_class_ids for d in nested)
        has_no_helmet = any(d.class_id in self._no_helmet_class_ids for d in nested)
        return has_rider and has_no_helmet

    def _max_no_helmet_conf(self, nested: list) -> float:
        scores = [d.confidence for d in nested if d.class_id in self._no_helmet_class_ids]
        return max(scores) if scores else 0.0

    def _extract_plate_roi(
        self, frame: Any, bike_crop: Any, bike_bbox: BBox
    ) -> tuple[BBox | None, Any]:
        x1, y1, _, _ = bike_bbox
        if self._plate is None:
            bh, bw = bike_crop.shape[:2]
            px1, px2 = int(bw * 0.25), int(bw * 0.75)
            py1, py2 = int(bh * 0.60), int(bh * 0.95)
            return (x1 + px1, y1 + py1, x1 + px2, y1 + py2), bike_crop[py1:py2, px1:px2]

        plates = self._plate.detect(bike_crop)
        if not plates:
            return None, bike_crop
        bh, bw = bike_crop.shape[:2]
        bx1, by1, bx2, by2 = _clip_bbox(*plates[0].bbox, bw, bh)
        if bx2 <= bx1 or by2 <= by1:
            return None, bike_crop
        return (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2), bike_crop[by1:by2, bx1:bx2]

    def _safe_anpr(self, plate_crop: Any) -> dict[str, str]:
        try:
            result = self._anpr_fn(plate_crop) or {}
            plate_number = str(result.get("plate_number", "UNKNOWN")).strip() or "UNKNOWN"
            confidence = str(result.get("confidence", "0.00")).strip() or "0.00"
            return {"plate_number": plate_number, "confidence": confidence}
        except Exception:
            return {"plate_number": "UNKNOWN", "confidence": "0.00"}

    def _is_duplicate_violation(self, bbox: BBox) -> bool:
        now = time.time()
        alive: list[tuple[BBox, float]] = []
        is_duplicate = False
        for old_bbox, ts in self._recent_fines:
            if (now - ts) <= self._fines_cooldown_sec:
                alive.append((old_bbox, ts))
                if _iou(bbox, old_bbox) >= 0.5:
                    is_duplicate = True
        self._recent_fines = alive
        return is_duplicate


def draw_helmet_violations(frame: Any, violations: list[HelmetViolation]) -> None:
    for v in violations:
        x1, y1, x2, y2 = v.bike_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"FINE ISSUED: {v.plate_number}"
        label_y = max(28, y1 - 10)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 12, label_y + 6), (22, 22, 22), -1)
        cv2.putText(
            frame, label, (x1 + 6, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA,
        )
