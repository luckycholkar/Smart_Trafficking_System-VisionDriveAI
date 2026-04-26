"""
Ultralytics YOLO implementation of the Detector interface.

Lazy-imports `ultralytics` so importing `visiondrive.models` does not require
the package to be installed for static analysis or test collection.
"""

from __future__ import annotations

from typing import Any, Iterable

from visiondrive.models.base import Detection, Detector, TrackedDetection
from visiondrive.models.registry import register_detector


class UltralyticsDetector(Detector):
    """Wraps `ultralytics.YOLO` to satisfy the Detector contract."""

    framework_name = "ultralytics"

    def __init__(
        self,
        weights_path: str,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO  # local import to keep module light

        self._model = YOLO(weights_path)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._tracker = tracker_config

    @property
    def framework(self) -> str:
        return self.framework_name

    def detect(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[Detection]:
        results = self._model.predict(
            frame,
            classes=list(classes) if classes else None,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        return self._unpack_detections(results[0])

    def track(self, frame: Any, *, classes: Iterable[int] | None = None) -> list[TrackedDetection]:
        results = self._model.track(
            frame,
            persist=True,
            classes=list(classes) if classes else None,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            tracker=self._tracker,
            verbose=False,
        )
        return self._unpack_tracked(results[0])

    @staticmethod
    def _unpack_detections(result: Any) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        out: list[Detection] = []
        for i, raw in enumerate(xyxy):
            x1, y1, x2, y2 = (int(v) for v in raw)
            out.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(cls[i]) if cls is not None else -1,
                    confidence=float(conf[i]) if conf is not None else 0.0,
                )
            )
        return out

    @staticmethod
    def _unpack_tracked(result: Any) -> list[TrackedDetection]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        out: list[TrackedDetection] = []
        for i, raw in enumerate(xyxy):
            x1, y1, x2, y2 = (int(v) for v in raw)
            out.append(
                TrackedDetection(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(cls[i]) if cls is not None else -1,
                    confidence=float(conf[i]) if conf is not None else 0.0,
                    track_id=int(ids[i]),
                )
            )
        return out


register_detector("ultralytics", UltralyticsDetector)
