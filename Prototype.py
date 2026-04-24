import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import os

# Prefer a stronger model for better two-wheeler recall.
MODEL_CANDIDATES = ["yolov8s.pt", "yolov8n.pt"]
model_path = next((m for m in MODEL_CANDIDATES if os.path.exists(m)), "yolov8n.pt")
model = YOLO(model_path)
print(f"Using model: {model_path}")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

# Include bicycle(1) + motorcycle(3) to improve two-wheeler detection.
vehicle_classes = [1, 2, 3, 5, 7]
vehicle_class_names = {1: "Cycle", 2: "Car", 3: "Bike/Scooter", 5: "Bus", 7: "Truck"}
TRACK_TTL_FRAMES = 18
COUNT_SMOOTH_WINDOW = 12
DETECTION_CONF = 0.15
DETECTION_IOU = 0.50
DETECTION_IMGSZ = 960
track_memory = {}
count_history = deque(maxlen=COUNT_SMOOTH_WINDOW)
frame_index = 0

# Frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS fallback
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# Video writer
out = cv2.VideoWriter(
    "cctv_output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

print("Processing started... Press 'q' to stop")


def draw_ui_header(frame, vehicle_count):
    """Draw a clean, real-world style info header."""
    header_h = 92
    panel_w = 500

    # translucent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (panel_w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # title and status
    cv2.putText(
        frame,
        "Smart Traffic Monitoring",
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "LIVE",
        (420, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 240, 120),
        2,
        cv2.LINE_AA
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        f"Time: {timestamp}",
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Vehicles In View: {vehicle_count}",
        (285, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 255),
        2,
        cv2.LINE_AA
    )


def draw_status_strip(frame, active_count, smoothed_count):
    """Draw modern bottom status bar for real-world monitoring."""
    h, w = frame.shape[:2]
    strip_top = h - 46
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, strip_top), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        f"Active Tracks: {active_count}",
        (18, h - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Smoothed Count: {smoothed_count:.1f}",
        (230, h - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 225, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "Press Q to exit",
        (w - 170, h - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (170, 170, 170),
        1,
        cv2.LINE_AA
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1

    # cleaner visualization than default model.plot()
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        verbose=False,
        conf=DETECTION_CONF,
        iou=DETECTION_IOU,
        imgsz=DETECTION_IMGSZ,
        agnostic_nms=True,
        max_det=300,
        tracker="bytetrack.yaml"
    )

    boxes = results[0].boxes
    seen_this_frame = set()

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i]
            obj_id = ids[i]
            cls = classes[i] if len(classes) > i else -1
            conf = confs[i] if len(confs) > i else 0.0

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            seen_this_frame.add(obj_id)
            track_memory[obj_id] = {
                "bbox": (x1, y1, x2, y2),
                "cls": cls,
                "conf": conf,
                "last_seen_frame": frame_index
            }

    stale_ids = []
    active_count = 0
    for obj_id, data in track_memory.items():
        age = frame_index - data["last_seen_frame"]
        if age > TRACK_TTL_FRAMES:
            stale_ids.append(obj_id)
            continue

        active_count += 1
        x1, y1, x2, y2 = data["bbox"]
        cls = data["cls"]
        conf = data["conf"]
        is_stale = obj_id not in seen_this_frame

        box_color = (90, 215, 255) if not is_stale else (125, 125, 125)
        thickness = 2 if not is_stale else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        cls_name = vehicle_class_names.get(cls, "Vehicle")
        label = f"{cls_name} #{obj_id}"
        label_y = max(24, y1 - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        chip_color = (24, 24, 24) if not is_stale else (45, 45, 45)
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 10, label_y + 3), chip_color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    for obj_id in stale_ids:
        track_memory.pop(obj_id, None)

    count_history.append(active_count)
    smoothed_count = sum(count_history) / max(1, len(count_history))
    draw_ui_header(frame, active_count)
    draw_status_strip(frame, active_count, smoothed_count)

    cv2.imshow("CCTV Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording saved as cctv_output.avi")