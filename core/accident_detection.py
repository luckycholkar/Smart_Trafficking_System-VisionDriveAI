from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

from core.notifier import EmergencyAlert

BBox = Tuple[int, int, int, int]


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
    involved_ids: List[int]
    crash_bbox: BBox
    payload: Dict[str, Any]


@dataclass
class AccidentDetectorConfig:
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
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None


@dataclass
class _TrackState:
    center: Tuple[float, float]
    prev_center: Optional[Tuple[float, float]] = None
    speed: float = 0.0
    prev_speed: float = 0.0
    last_frame: int = 0


@dataclass
class _PairState:
    first_frame: int
    last_frame: int
    first_center: Tuple[float, float]
    last_center: Tuple[float, float]
    merged_bbox: BBox


def _center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _merge_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _estimate_severity(max_speed_drop: float, crash_iou: float) -> str:
    score = (max_speed_drop * 0.7) + (crash_iou * 30.0)
    if score >= 35:
        return "HIGH"
    if score >= 20:
        return "MEDIUM"
    return "LOW"


class AccidentDetector:
    """
    Detects potential accidents from tracked vehicle detections.

    Confirmation signals:
    1) Significant overlap (IoU > threshold) + merged region stationary > X sec.
    2) Sudden speed drop from high motion to near-zero in 1-2 frames.
    """

    def __init__(
        self,
        config: Optional[AccidentDetectorConfig] = None,
        emergency_alert: Optional[EmergencyAlert] = None,
    ) -> None:
        self.config = config or AccidentDetectorConfig()
        self.tracks: Dict[int, _TrackState] = {}
        self.pairs: Dict[Tuple[int, int], _PairState] = {}
        self.recent_events: Dict[str, int] = {}
        self.emergency_alert = emergency_alert
        Path(self.config.incidents_dir).mkdir(parents=True, exist_ok=True)

    def update(
        self,
        frame: Any,
        frame_index: int,
        detections: List[VehicleDetection],
    ) -> List[AccidentEvent]:
        events: List[AccidentEvent] = []
        if not detections:
            return events

        self._update_tracks(frame_index, detections)
        events.extend(self._check_impact_events(frame, frame_index, detections))
        events.extend(self._check_collision_stationary_events(frame, frame_index, detections))
        if self.emergency_alert:
            for event in events:
                self.emergency_alert.notify_async(event.payload)
        return events

    def _update_tracks(self, frame_index: int, detections: List[VehicleDetection]) -> None:
        for d in detections:
            c = _center(d.bbox)
            prev = self.tracks.get(d.object_id)
            if prev is None:
                self.tracks[d.object_id] = _TrackState(center=c, last_frame=frame_index)
                continue
            speed = _distance(c, prev.center)
            self.tracks[d.object_id] = _TrackState(
                center=c,
                prev_center=prev.center,
                speed=speed,
                prev_speed=prev.speed,
                last_frame=frame_index,
            )

    def _check_impact_events(
        self, frame: Any, frame_index: int, detections: List[VehicleDetection]
    ) -> List[AccidentEvent]:
        events: List[AccidentEvent] = []
        by_id = {d.object_id: d for d in detections}
        for object_id, track in self.tracks.items():
            if object_id not in by_id:
                continue
            prev_fast = track.prev_speed >= self.config.high_speed_px_per_frame
            now_near_zero = track.speed <= self.config.near_zero_speed_px_per_frame
            if not (prev_fast and now_near_zero):
                continue
            # Frame continuity check avoids stale comparisons.
            if (frame_index - track.last_frame) > self.config.speed_drop_window_frames:
                continue

            bbox = by_id[object_id].bbox
            severity = _estimate_severity(track.prev_speed - track.speed, 0.0)
            event = self._build_event(
                frame=frame,
                frame_index=frame_index,
                crash_bbox=bbox,
                involved_ids=[object_id],
                trigger_type="IMPACT_ALERT",
                severity=severity,
            )
            if event:
                events.append(event)
        return events

    def _check_collision_stationary_events(
        self, frame: Any, frame_index: int, detections: List[VehicleDetection]
    ) -> List[AccidentEvent]:
        events: List[AccidentEvent] = []
        n = len(detections)
        min_stationary_frames = int(self.config.stationary_seconds * self.config.fps)
        for i in range(n):
            for j in range(i + 1, n):
                a = detections[i]
                b = detections[j]
                overlap = _iou(a.bbox, b.bbox)
                if overlap < self.config.overlap_iou_threshold:
                    continue

                pair_key = tuple(sorted((a.object_id, b.object_id)))
                merged = _merge_bbox(a.bbox, b.bbox)
                c = _center(merged)
                state = self.pairs.get(pair_key)
                if state is None:
                    self.pairs[pair_key] = _PairState(
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
                is_stationary = stationary_disp <= self.config.stationary_pixel_tolerance
                is_persistent = elapsed >= min_stationary_frames
                if not (is_stationary and is_persistent):
                    continue

                drop_a = max(0.0, self.tracks.get(a.object_id, _TrackState(c)).prev_speed - self.tracks.get(a.object_id, _TrackState(c)).speed)
                drop_b = max(0.0, self.tracks.get(b.object_id, _TrackState(c)).prev_speed - self.tracks.get(b.object_id, _TrackState(c)).speed)
                severity = _estimate_severity(max(drop_a, drop_b), overlap)
                event = self._build_event(
                    frame=frame,
                    frame_index=frame_index,
                    crash_bbox=merged,
                    involved_ids=[a.object_id, b.object_id],
                    trigger_type="COLLISION_STATIONARY",
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
        involved_ids: List[int],
        trigger_type: str,
        severity: str,
    ) -> Optional[AccidentEvent]:
        # Avoid duplicate notifications for same region in nearby frames.
        event_key = f"{trigger_type}:{crash_bbox[0]//20}:{crash_bbox[1]//20}:{crash_bbox[2]//20}:{crash_bbox[3]//20}"
        last_frame = self.recent_events.get(event_key)
        if last_frame is not None and (frame_index - last_frame) < int(self.config.fps * 4):
            return None
        self.recent_events[event_key] = frame_index

        timestamp = datetime.now().isoformat(timespec="seconds")
        snapshot_url = self._save_snapshot(frame, frame_index)
        location = self.config.location_name
        if self.config.gps_lat is not None and self.config.gps_lon is not None:
            location = f"{self.config.location_name} ({self.config.gps_lat},{self.config.gps_lon})"

        payload = {
            "timestamp": timestamp,
            "location": location,
            "severity_level": severity,
            "snapshot_url": snapshot_url,
            "camera_id": self.config.camera_id,
            "trigger_type": trigger_type,
            "involved_vehicle_ids": involved_ids,
            "gps_lat": self.config.gps_lat,
            "gps_lon": self.config.gps_lon,
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
        full_path = Path(self.config.incidents_dir) / filename
        cv2.imwrite(str(full_path), frame)
        return str(full_path)


def simulate_emergency_alert_handler(
    event: AccidentEvent,
    dashboard_sender: Optional[Callable[[Dict[str, Any]], bool]] = None,
    external_sender: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """
    Simulates sending the accident payload to:
    1) Internal handler dashboard
    2) External emergency channel (SMS/Email/API)
    """
    payload = event.payload
    dashboard_ok = dashboard_sender(payload) if dashboard_sender else True
    external_ok = external_sender(payload) if external_sender else True

    result = {
        "event_id": event.event_id,
        "dashboard_sent": dashboard_ok,
        "external_alert_sent": external_ok,
        "dispatch_status": "CONTACTING_EMERGENCY_SERVICES" if (dashboard_ok and external_ok) else "RETRY_REQUIRED",
        "payload": payload,
    }
    print("[ACCIDENT ALERT]", result)
    return result


def draw_accident_alert_overlay(
    frame: Any,
    frame_index: int,
    events: List[AccidentEvent],
    flash_every_n_frames: int = 12,
) -> None:
    """
    Draws flashing red border + prominent emergency message on detection.
    """
    if not events:
        return
    should_flash = (frame_index // max(1, flash_every_n_frames)) % 2 == 0
    if not should_flash:
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
    box_y1 = 18
    box_y2 = 18 + th + 18
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
    cv2.putText(
        frame,
        msg,
        (box_x1 + pad, box_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
