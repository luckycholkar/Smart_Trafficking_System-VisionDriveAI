from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

from stubs.anpr_stub import run_anpr


BBox = Tuple[int, int, int, int]


@dataclass
class HelmetViolation:
    bike_bbox: BBox
    plate_bbox: Optional[BBox]
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


class HelmetViolationPipeline:
    """
    Two-stage violation detection:
    1) Detect two-wheelers from primary model.
    2) Run nested helmet model inside each two-wheeler crop.
    3) On no-helmet, run ANPR on plate ROI and issue fine payload.
    """

    def __init__(
        self,
        vehicle_model_path: str = "yolov8n.pt",
        helmet_model_path: str = "helmet_detector.pt",
        plate_model_path: Optional[str] = None,
        motorcycle_class_id: int = 3,
        rider_class_ids: Sequence[int] = (0,),
        no_helmet_class_ids: Sequence[int] = (2,),
        bike_conf: float = 0.20,
        helmet_conf: float = 0.35,
        no_helmet_min_conf: float = 0.45,
        fine_cooldown_sec: float = 5.0,
        anpr_fn: Callable[[Any], Dict[str, str]] = run_anpr,
    ) -> None:
        self.vehicle_model = YOLO(vehicle_model_path)
        self.helmet_model = YOLO(helmet_model_path)
        self.plate_model = YOLO(plate_model_path) if plate_model_path else None

        self.motorcycle_class_id = motorcycle_class_id
        self.rider_class_ids = set(rider_class_ids)
        self.no_helmet_class_ids = set(no_helmet_class_ids)
        self.bike_conf = bike_conf
        self.helmet_conf = helmet_conf
        self.no_helmet_min_conf = no_helmet_min_conf
        self.fine_cooldown_sec = fine_cooldown_sec
        self.anpr_fn = anpr_fn
        self._recent_fines: List[Tuple[BBox, float]] = []

    def detect_violations(self, frame) -> List[HelmetViolation]:
        h, w = frame.shape[:2]
        violations: List[HelmetViolation] = []

        bike_result = self.vehicle_model.predict(
            frame,
            classes=[self.motorcycle_class_id],
            conf=self.bike_conf,
            verbose=False,
        )[0]

        boxes = bike_result.boxes
        if boxes is None or boxes.xyxy is None:
            return violations

        bike_xyxy = boxes.xyxy.cpu().numpy()
        bike_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []

        for i, raw_box in enumerate(bike_xyxy):
            x1, y1, x2, y2 = _clip_bbox(raw_box[0], raw_box[1], raw_box[2], raw_box[3], w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            bike_crop = frame[y1:y2, x1:x2]
            if bike_crop.size == 0:
                continue

            nested = self.helmet_model.predict(bike_crop, conf=self.helmet_conf, verbose=False)[0]
            if not self._is_no_helmet(nested):
                continue

            no_helmet_score = self._max_no_helmet_conf(nested)
            if no_helmet_score < self.no_helmet_min_conf:
                continue
            if self._is_duplicate_violation((x1, y1, x2, y2)):
                continue
            plate_bbox_global, plate_crop = self._extract_plate_roi(frame, bike_crop, (x1, y1, x2, y2))
            plate_info = self._run_safe_anpr(plate_crop)
            plate_number = plate_info.get("plate_number", "UNKNOWN")
            anpr_conf = plate_info.get("confidence", "0.00")

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._recent_fines.append(((x1, y1, x2, y2), time.time()))
            violations.append(
                HelmetViolation(
                    bike_bbox=(x1, y1, x2, y2),
                    plate_bbox=plate_bbox_global,
                    bike_conf=float(bike_conf[i]) if i < len(bike_conf) else 0.0,
                    no_helmet_conf=no_helmet_score,
                    plate_number=plate_number,
                    anpr_confidence=anpr_conf,
                    fine_id=f"FINE-{datetime.now().strftime('%H%M%S%f')[-8:]}",
                    timestamp=ts,
                )
            )

        return violations

    def _is_no_helmet(self, nested_result) -> bool:
        boxes = nested_result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return False
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        has_rider = any(cls in self.rider_class_ids for cls in classes)
        has_no_helmet = any(
            (cls in self.no_helmet_class_ids) and (conf >= self.helmet_conf)
            for cls, conf in zip(classes, confs)
        )
        return has_rider and has_no_helmet

    def _max_no_helmet_conf(self, nested_result) -> float:
        boxes = nested_result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return 0.0
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        scores = [float(conf) for cls, conf in zip(classes, confs) if cls in self.no_helmet_class_ids]
        return max(scores) if scores else 0.0

    def _extract_plate_roi(self, frame, bike_crop, bike_bbox: BBox) -> Tuple[Optional[BBox], Any]:
        x1, y1, _, _ = bike_bbox
        if self.plate_model is None:
            # Fallback: lower-center region of bike for ANPR.
            bh, bw = bike_crop.shape[:2]
            px1 = int(bw * 0.25)
            px2 = int(bw * 0.75)
            py1 = int(bh * 0.60)
            py2 = int(bh * 0.95)
            plate_crop = bike_crop[py1:py2, px1:px2]
            return (x1 + px1, y1 + py1, x1 + px2, y1 + py2), plate_crop

        plate_result = self.plate_model.predict(bike_crop, conf=0.25, verbose=False)[0]
        pboxes = plate_result.boxes
        if pboxes is None or pboxes.xyxy is None or len(pboxes.xyxy) == 0:
            return None, bike_crop

        pxyxy = pboxes.xyxy.cpu().numpy()[0]
        bx1, by1, bx2, by2 = map(int, pxyxy)
        bh, bw = bike_crop.shape[:2]
        bx1, by1, bx2, by2 = _clip_bbox(bx1, by1, bx2, by2, bw, bh)
        if bx2 <= bx1 or by2 <= by1:
            return None, bike_crop

        plate_crop = bike_crop[by1:by2, bx1:bx2]
        return (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2), plate_crop

    def _run_safe_anpr(self, plate_crop: Any) -> Dict[str, str]:
        try:
            result = self.anpr_fn(plate_crop) or {}
            plate_number = str(result.get("plate_number", "UNKNOWN")).strip() or "UNKNOWN"
            confidence = str(result.get("confidence", "0.00")).strip() or "0.00"
            return {"plate_number": plate_number, "confidence": confidence}
        except Exception:
            return {"plate_number": "UNKNOWN", "confidence": "0.00"}

    def _is_duplicate_violation(self, bbox: BBox) -> bool:
        now = time.time()
        alive: List[Tuple[BBox, float]] = []
        is_duplicate = False
        for old_bbox, ts in self._recent_fines:
            if (now - ts) <= self.fine_cooldown_sec:
                alive.append((old_bbox, ts))
                if _iou(bbox, old_bbox) >= 0.5:
                    is_duplicate = True
        self._recent_fines = alive
        return is_duplicate


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(a_area + b_area - inter_area)


def draw_clean_helmet_ui(frame, violations: List[HelmetViolation]) -> None:
    """
    Draw only helmet-rule violations for uncluttered video output.
    """
    for v in violations:
        x1, y1, x2, y2 = v.bike_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"FINE ISSUED: {v.plate_number}"
        label_y = max(28, y1 - 10)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)

        cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 12, label_y + 6), (22, 22, 22), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
