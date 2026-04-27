"""
VisionPipeline: threaded reader + inference loop.

Producer for the DataBus. Frame-source-agnostic detection happens via the
Detector interface (visiondrive.models.Detector) so swapping frameworks is a
config change, not a code change.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from visiondrive.constants import (
    BUS_CONGESTION,
    BUS_FPS,
    BUS_FRAME_INDEX,
    BUS_LANE_COUNTS,
    BUS_SIGNAL_STATE,
    BUS_VEHICLE_COUNT,
)
from visiondrive.core.data_bus import DataBus
from visiondrive.core.overlays import (
    class_label,
    congestion_score,
    draw_congestion_caption,
    draw_lane_signal_overlay,
    draw_stop_line,
    draw_vehicle_box,
    lane_for_x,
)
from visiondrive.models import Detector
from visiondrive.settings import Settings
from visiondrive.violations.red_light import RedLightViolationDetector

log = logging.getLogger(__name__)

QUEUE_SIZE = 6
QUEUE_PUT_TIMEOUT_SEC = 0.1
QUEUE_GET_TIMEOUT_SEC = 0.2


class VisionPipeline:
    """
    Frame-reader thread feeds a queue; inference thread consumes it. Drops
    frames under back-pressure to keep the demo realtime.
    """

    def __init__(
        self,
        settings: Settings,
        detector: Detector,
        data_bus: DataBus,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._data_bus = data_bus
        self._frame_queue: queue.Queue[tuple[int, Any]] = queue.Queue(maxsize=QUEUE_SIZE)
        self.stop_event = threading.Event()

        video_path = settings.resolve_path(settings.video.path)
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}. "
                "Place a video at the configured path or update settings.video.path."
            )
        self._video_path = str(video_path)
        self._output_path = str(settings.resolve_path(settings.video.output_path))

    def run(self) -> None:
        reader = threading.Thread(target=self._reader_loop, name="vd-reader", daemon=True)
        infer = threading.Thread(target=self._infer_loop, name="vd-infer", daemon=True)
        reader.start()
        infer.start()
        reader.join()
        infer.join()

    def _reader_loop(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        idx = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                try:
                    self._frame_queue.put((idx, frame), timeout=QUEUE_PUT_TIMEOUT_SEC)
                except queue.Full:
                    continue  # drop frame to stay realtime
        finally:
            cap.release()
            self.stop_event.set()

    def _infer_loop(self) -> None:
        s = self._settings
        violation_detector: RedLightViolationDetector | None = None
        output: cv2.VideoWriter | None = None
        start = time.time()
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)

        while not self.stop_event.is_set() or not self._frame_queue.empty():
            try:
                frame_idx, frame = self._frame_queue.get(timeout=QUEUE_GET_TIMEOUT_SEC)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            if violation_detector is None:
                stop_line_y = int(h * s.detection.stop_line_ratio_y)
                violation_detector = RedLightViolationDetector(stop_line_y=stop_line_y)
                output = cv2.VideoWriter(
                    self._output_path,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    s.video.output_fps,
                    (w, h),
                )

            tracked = self._detector.track(frame, classes=s.detection.vehicle_class_ids)
            lane_counts: dict[str, int] = {lane: 0 for lane in s.lanes.ids}
            signal_state = self._data_bus.snapshot().get(BUS_SIGNAL_STATE, {})

            for det in tracked:
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                lane = lane_for_x(cx, w, s.lanes.ids)
                lane_counts[lane] += 1

                violation = violation_detector.update(
                    object_id=det.track_id,
                    center_y=cy,
                    lane_id=lane,
                    signal_map=signal_state,
                    frame_index=frame_idx,
                )
                if violation:
                    self._data_bus.append_violation(violation)

                draw_vehicle_box(
                    frame,
                    det.bbox,
                    f"{class_label(det.class_id)} | {lane} #{det.track_id}",
                )

            draw_lane_signal_overlay(frame, s.lanes.ids, lane_counts, signal_state)
            draw_stop_line(frame, violation_detector.stop_line_y)
            elapsed = max(1e-6, time.time() - start)
            fps = frame_idx / elapsed
            score = congestion_score(lane_counts, s.detection.saturation_per_lane)
            draw_congestion_caption(frame, score)

            self._data_bus.update(
                **{
                    BUS_FRAME_INDEX: frame_idx,
                    BUS_VEHICLE_COUNT: sum(lane_counts.values()),
                    BUS_LANE_COUNTS: lane_counts,
                    BUS_CONGESTION: score,
                    BUS_FPS: round(fps, 2),
                }
            )

            if output is not None:
                output.write(frame)
            if s.features.display_window:
                cv2.imshow("VisionDrive AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()

        if output is not None:
            output.release()
        if s.features.display_window:
            cv2.destroyAllWindows()
        log.info("Vision pipeline stopped at frame %s", self._data_bus.snapshot().get(BUS_FRAME_INDEX))
