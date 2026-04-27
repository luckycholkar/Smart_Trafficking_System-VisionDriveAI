"""OpenCV drawing helpers used by the vision pipeline. Pure presentation."""

from __future__ import annotations

from typing import Any

import cv2

from visiondrive.constants import COCO_CLASS_NAMES, SIGNAL_GREEN


def lane_for_x(x: float, width: int, lanes: list[str]) -> str:
    lane_width = max(1, width // len(lanes))
    idx = min(len(lanes) - 1, int(x // lane_width))
    return lanes[idx]


def congestion_score(lane_counts: dict[str, int], saturation_per_lane: int) -> float:
    if not lane_counts:
        return 0.0
    sat = max(1, saturation_per_lane)
    normalized = [min(c / sat, 1.0) for c in lane_counts.values()]
    return round((sum(normalized) / len(normalized)) * 100.0, 2)


def class_label(class_id: int) -> str:
    return COCO_CLASS_NAMES.get(class_id, "Vehicle")


def draw_lane_signal_overlay(
    frame: Any,
    lanes: list[str],
    lane_counts: dict[str, int],
    signal_state: dict[str, str],
) -> None:
    h, w = frame.shape[:2]
    lane_width = max(1, w // len(lanes))
    overlay = frame.copy()

    for idx, lane in enumerate(lanes):
        x1 = idx * lane_width
        x2 = w if idx == len(lanes) - 1 else (idx + 1) * lane_width
        state = signal_state.get(lane, "RED")
        count = lane_counts.get(lane, 0)
        color = (40, 180, 40) if state == SIGNAL_GREEN else (30, 30, 210)

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


def draw_vehicle_box(frame: Any, bbox: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_stop_line(frame: Any, stop_line_y: int) -> None:
    h, w = frame.shape[:2]
    _ = h
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 2)


def draw_congestion_caption(frame: Any, score: float) -> None:
    h = frame.shape[0]
    cv2.putText(
        frame,
        f"Congestion: {score:.1f}%",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
