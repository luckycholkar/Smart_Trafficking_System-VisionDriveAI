"""
Minimal standalone demo: detector -> annotated video, no API/dashboard.

Useful for iterating on the detection layer in isolation. Run from the
repo root:

    python examples/prototype_demo.py
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visiondrive.constants import COCO_CLASS_NAMES
from visiondrive.models import build_detector
from visiondrive.settings import get_settings

SMOOTH_WINDOW = 12


def main() -> int:
    settings = get_settings()
    detector = build_detector(settings.model)

    video_path = settings.resolve_path(settings.video.path)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = settings.resolve_path("data/videos/prototype_output.avi")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
    )

    smoothed: deque[int] = deque(maxlen=SMOOTH_WINDOW)
    print(f"Using detector: {detector.framework}")
    print("Press 'q' to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracked = detector.track(frame, classes=settings.detection.vehicle_class_ids)
            for det in tracked:
                x1, y1, x2, y2 = det.bbox
                cls_name = COCO_CLASS_NAMES.get(det.class_id, "Vehicle")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 215, 255), 2)
                cv2.putText(
                    frame, f"{cls_name} #{det.track_id}", (x1 + 4, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA,
                )

            smoothed.append(len(tracked))
            avg = sum(smoothed) / max(1, len(smoothed))
            cv2.putText(
                frame,
                f"{datetime.now():%H:%M:%S}  vehicles={len(tracked)}  smoothed={avg:.1f}",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

            writer.write(frame)
            if settings.features.display_window:
                cv2.imshow("VisionDrive prototype", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if settings.features.display_window:
            cv2.destroyAllWindows()
        print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
