"""
Heuristic accident detector.

Operates on tracked detections from the main pipeline (no ML model of its
own). Two confirmation signals:
  1. Significant overlap (IoU > threshold) plus merged region stationary > N sec.
  2. Sudden speed drop from high motion to near-zero in 1-2 frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from visiondrive.constants import EVENT_ACCIDENT_COLLISION, EVENT_ACCIDENT_IMPACT
from visiondrive.notifications.twilio_sms import EmergencyAlert
from visiondrive.settings import AccidentConfig

BBox = tuple[int, int, int, int]


@dataclass
class VehicleDetection:
    object_id: int
    bbox: BBox


@dataclass
class AccidentEvent:
    event_id: str
    timestamp: str
    location: str
    severity_level: str
    snapshot_url: str
    trigger_type: str
    involved_ids: list[int]
    crash_bbox: BBox
    payload: dict[str, Any]


@dataclass
class _TrackState:
    center: tuple[float, float]
    prev_center: tuple[float, float] | None = None
    speed: float = 0.0
    prev_speed: float = 0.0
    last_frame: int = 0


@dataclass
class _PairState:
    first_frame: int
    last_frame: int
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    merged_bbox: BBox


def _center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _merge_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _iou(a: BBox, b: BBox) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(a_area + b_area - inter)


def _estimate_severity(max_speed_drop: float, crash_iou: float) -> str:
    score = (max_speed_drop * 0.7) + (crash_iou * 30.0)
    if score >= 35:
        return "HIGH"
    if score >= 20:
        return "MEDIUM"
    return "LOW"


class AccidentDetector:
    def __init__(
        self,
        config: AccidentConfig,
        emergency_alert: EmergencyAlert | None = None,
        *,
        incidents_root: Path | None = None,
    ) -> None:
        self.config = config
        self._tracks: dict[int, _TrackState] = {}
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self._recent_events: dict[str, int] = {}
        self._emergency_alert = emergency_alert
        self._incidents_dir = (incidents_root or Path.cwd()) / config.incidents_dir
        self._incidents_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        frame: Any,
        frame_index: int,
        detections: list[VehicleDetection],
    ) -> list[AccidentEvent]:
        if not detections:
            return []
        events: list[AccidentEvent] = []
        self._update_tracks(frame_index, detections)
        events.extend(self._check_impact_events(frame, frame_index, detections))
        events.extend(self._check_collision_events(frame, frame_index, detections))
        if self._emergency_alert:
            for event in events:
                self._emergency_alert.notify_async(event.payload)
        return events

    def _update_tracks(self, frame_index: int, detections: list[VehicleDetection]) -> None:
        for d in detections:
            c = _center(d.bbox)
            prev = self._tracks.get(d.object_id)
            if prev is None:
                self._tracks[d.object_id] = _TrackState(center=c, last_frame=frame_index)
                continue
            speed = _distance(c, prev.center)
            self._tracks[d.object_id] = _TrackState(
                center=c,
                prev_center=prev.center,
                speed=speed,
                prev_speed=prev.speed,
                last_frame=frame_index,
            )

    def _check_impact_events(
        self, frame: Any, frame_index: int, detections: list[VehicleDetection]
    ) -> list[AccidentEvent]:
        events: list[AccidentEvent] = []
        by_id = {d.object_id: d for d in detections}
        cfg = self.config
        for object_id, track in self._tracks.items():
            if object_id not in by_id:
                continue
            prev_fast = track.prev_speed >= cfg.high_speed_px_per_frame
            now_near_zero = track.speed <= cfg.near_zero_speed_px_per_frame
            if not (prev_fast and now_near_zero):
                continue
            if (frame_index - track.last_frame) > cfg.speed_drop_window_frames:
                continue

            bbox = by_id[object_id].bbox
            severity = _estimate_severity(track.prev_speed - track.speed, 0.0)
            event = self._build_event(
                frame=frame,
                frame_index=frame_index,
                crash_bbox=bbox,
                involved_ids=[object_id],
                trigger_type=EVENT_ACCIDENT_IMPACT,
                severity=severity,
            )
            if event:
                events.append(event)
        return events

    def _check_collision_events(
        self, frame: Any, frame_index: int, detections: list[VehicleDetection]
    ) -> list[AccidentEvent]:
        events: list[AccidentEvent] = []
        cfg = self.config
        n = len(detections)
        min_stationary_frames = int(cfg.stationary_seconds * cfg.fps)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = detections[i], detections[j]
                overlap = _iou(a.bbox, b.bbox)
                if overlap < cfg.overlap_iou_threshold:
                    continue

                pair_key = tuple(sorted((a.object_id, b.object_id)))
                merged = _merge_bbox(a.bbox, b.bbox)
                c = _center(merged)
                state = self._pairs.get(pair_key)
                if state is None:
                    self._pairs[pair_key] = _PairState(
                        first_frame=frame_index,
                        last_frame=frame_index,
                        first_center=c,
                        last_center=c,
                        merged_bbox=merged,
                    )
                    continue

                state.last_frame = frame_index
                state.last_center = c
                state.merged_bbox = merged
                stationary_disp = _distance(state.first_center, state.last_center)
                elapsed = state.last_frame - state.first_frame
                if not (
                    stationary_disp <= cfg.stationary_pixel_tolerance
                    and elapsed >= min_stationary_frames
                ):
                    continue

                drop_a = max(
                    0.0,
                    self._tracks.get(a.object_id, _TrackState(c)).prev_speed
                    - self._tracks.get(a.object_id, _TrackState(c)).speed,
                )
                drop_b = max(
                    0.0,
                    self._tracks.get(b.object_id, _TrackState(c)).prev_speed
                    - self._tracks.get(b.object_id, _TrackState(c)).speed,
                )
                severity = _estimate_severity(max(drop_a, drop_b), overlap)
                event = self._build_event(
                    frame=frame,
                    frame_index=frame_index,
                    crash_bbox=merged,
                    involved_ids=[a.object_id, b.object_id],
                    trigger_type=EVENT_ACCIDENT_COLLISION,
                    severity=severity,
                )
                if event:
                    events.append(event)
        return events

    def _build_event(
        self,
        frame: Any,
        frame_index: int,
        crash_bbox: BBox,
        involved_ids: list[int],
        trigger_type: str,
        severity: str,
    ) -> AccidentEvent | None:
        cfg = self.config
        # Squelch duplicate notifications for nearby events in the same region.
        event_key = (
            f"{trigger_type}:{crash_bbox[0]//20}:{crash_bbox[1]//20}:"
            f"{crash_bbox[2]//20}:{crash_bbox[3]//20}"
        )
        last_frame = self._recent_events.get(event_key)
        if last_frame is not None and (frame_index - last_frame) < int(cfg.fps * 4):
            return None
        self._recent_events[event_key] = frame_index

        timestamp = datetime.now().isoformat(timespec="seconds")
        snapshot_url = self._save_snapshot(frame, frame_index)
        location = cfg.location_name
        if cfg.gps_lat is not None and cfg.gps_lon is not None:
            location = f"{cfg.location_name} ({cfg.gps_lat},{cfg.gps_lon})"

        payload = {
            "timestamp": timestamp,
            "location": location,
            "severity_level": severity,
            "snapshot_url": snapshot_url,
            "camera_id": cfg.camera_id,
            "trigger_type": trigger_type,
            "involved_vehicle_ids": involved_ids,
            "gps_lat": cfg.gps_lat,
            "gps_lon": cfg.gps_lon,
        }
        event_id = f"ACC-{datetime.now().strftime('%Y%m%d%H%M%S%f')[-10:]}"
        return AccidentEvent(
            event_id=event_id,
            timestamp=timestamp,
            location=location,
            severity_level=severity,
            snapshot_url=snapshot_url,
            trigger_type=trigger_type,
            involved_ids=involved_ids,
            crash_bbox=crash_bbox,
            payload=payload,
        )

    def _save_snapshot(self, frame: Any, frame_index: int) -> str:
        filename = f"accident_{frame_index}_{datetime.now().strftime('%H%M%S')}.jpg"
        path = self._incidents_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)


def draw_accident_alert(
    frame: Any,
    frame_index: int,
    events: list[AccidentEvent],
    *,
    flash_every_n_frames: int = 12,
    sender: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if not events:
        return
    if (frame_index // max(1, flash_every_n_frames)) % 2 != 0:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 8)
    primary = events[0]
    x1, y1, x2, y2 = primary.crash_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    msg = "ACCIDENT DETECTED: CONTACTING EMERGENCY SERVICES"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    pad = 10
    box_x1 = max(8, (w - tw) // 2 - pad)
    box_x2 = min(w - 8, (w + tw) // 2 + pad)
    box_y1, box_y2 = 18, 18 + th + 18
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
    cv2.putText(
        frame, msg, (box_x1 + pad, box_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if sender is not None:
        for event in events:
            sender(event.payload)
