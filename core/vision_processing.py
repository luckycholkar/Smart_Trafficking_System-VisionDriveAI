from __future__ import annotations

import queue
import threading
import time
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO

from config.settings import (
    DETECTION_CONF,
    DETECTION_IMGSZ,
    DETECTION_IOU,
    DISPLAY_WINDOW,
    LANES,
    SATURATION_PER_LANE,
    STOP_LINE_RATIO_Y,
    TRACKER_CONFIG,
    VEHICLE_CLASSES,
    VEHICLE_CLASS_NAMES,
)
from core.violation_detection import RedLightViolationDetector
from services.data_bus import DataBus


def _lane_for_x(x: float, width: int) -> str:
    lane_width = max(1, width // len(LANES))
    idx = min(len(LANES) - 1, int(x // lane_width))
    return LANES[idx]


def _congestion_score(lane_counts: Dict[str, int]) -> float:
    if not lane_counts:
        return 0.0
    normalized = [min(count / max(1, SATURATION_PER_LANE), 1.0) for count in lane_counts.values()]
    return round((sum(normalized) / len(normalized)) * 100.0, 2)


def _draw_lane_signal_overlay(frame, lane_counts: Dict[str, int], signal_state: Dict[str, str]) -> None:
    h, w = frame.shape[:2]
    lane_width = max(1, w // len(LANES))
    overlay = frame.copy()

    for idx, lane in enumerate(LANES):
        x1 = idx * lane_width
        x2 = w if idx == len(LANES) - 1 else (idx + 1) * lane_width
        state = signal_state.get(lane, "RED")
        count = lane_counts.get(lane, 0)
        color = (40, 180, 40) if state == "GREEN" else (30, 30, 210)

        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
        cv2.line(frame, (x1, 0), (x1, h), (190, 190, 190), 1)
        cv2.putText(
            frame,
            f"{lane} | {state} | {count}",
            (x1 + 8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


class VisionPipeline:
    """
    Threaded reader/inference pipeline for smooth hackathon demos.
    """

    def __init__(self, video_path: str, model_path: str, data_bus: DataBus):
        self.video_path = video_path
        self.data_bus = data_bus
        self.model = YOLO(model_path)
        self.frame_queue: "queue.Queue[Tuple[int, any]]" = queue.Queue(maxsize=6)
        self.stop_event = threading.Event()

    def run(self) -> None:
        reader = threading.Thread(target=self._reader_loop, daemon=True)
        infer = threading.Thread(target=self._infer_loop, daemon=True)
        reader.start()
        infer.start()
        reader.join()
        infer.join()

    def _reader_loop(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            try:
                self.frame_queue.put((idx, frame), timeout=0.1)
            except queue.Full:
                # Drop frame to keep realtime responsiveness.
                continue
        cap.release()
        self.stop_event.set()

    def _infer_loop(self) -> None:
        violation_detector = None
        output = None
        start = time.time()

        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame_idx, frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            if violation_detector is None:
                stop_line_y = int(h * STOP_LINE_RATIO_Y)
                violation_detector = RedLightViolationDetector(stop_line_y=stop_line_y)
                output = cv2.VideoWriter(
                    "hackathon_demo_output.avi",
                    cv2.VideoWriter_fourcc(*"XVID"),
                    25.0,
                    (w, h),
                )

            results = self.model.track(
                frame,
                persist=True,
                classes=VEHICLE_CLASSES,
                verbose=False,
                conf=DETECTION_CONF,
                iou=DETECTION_IOU,
                imgsz=DETECTION_IMGSZ,
                tracker=TRACKER_CONFIG,
            )

            lane_counts: Dict[str, int] = {lane: 0 for lane in LANES}
            signal_state = self.data_bus.snapshot().get("signal_state", {})

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
                for i, obj_id in enumerate(ids):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = int(classes[i]) if classes is not None and i < len(classes) else -1
                    cls_name = VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle")
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    lane = _lane_for_x(cx, w)
                    lane_counts[lane] += 1

                    vio = violation_detector.update(
                        object_id=int(obj_id),
                        center_y=cy,
                        lane_id=lane,
                        signal_map=signal_state,
                        frame_index=frame_idx,
                    )
                    if vio:
                        self.data_bus.append_violation(vio)

                    color = (0, 180, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        frame,
                        f"{cls_name} | {lane} #{obj_id}",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            _draw_lane_signal_overlay(frame, lane_counts, signal_state)
            cv2.line(frame, (0, violation_detector.stop_line_y), (w, violation_detector.stop_line_y), (0, 0, 255), 2)
            elapsed = max(1e-6, time.time() - start)
            fps = frame_idx / elapsed
            vehicle_count = sum(lane_counts.values())
            congestion_score = _congestion_score(lane_counts)
            cv2.putText(
                frame,
                f"Congestion: {congestion_score:.1f}%",
                (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.data_bus.update(
                frame_index=frame_idx,
                vehicle_count=vehicle_count,
                lane_counts=lane_counts,
                congestion_score=congestion_score,
                fps=round(fps, 2),
            )

            if output is not None:
                output.write(frame)
            if DISPLAY_WINDOW:
                cv2.imshow("Smart Traffic Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()

        if output is not None:
            output.release()
        cv2.destroyAllWindows()

