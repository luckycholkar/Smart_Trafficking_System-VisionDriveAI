VIDEO_PATH = "traffic.mp4"
MODEL_PATH = "yolov8n.pt"

LANES = ["lane_1", "lane_2", "lane_3", "lane_4"]
# COCO classes:
# 1=bicycle, 2=car, 3=motorcycle (use this for scooters too), 5=bus, 7=truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {
    1: "Cycle",
    2: "Car",
    3: "Bike/Scooter",
    5: "Bus",
    7: "Truck",
}
TRACKER_CONFIG = "bytetrack.yaml"
DETECTION_CONF = 0.15
DETECTION_IOU = 0.45
DETECTION_IMGSZ = 960

# Relative stop line location measured from top of frame.
STOP_LINE_RATIO_Y = 0.62

# Dashboard / runtime tuning
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
SIGNAL_RECOMPUTE_INTERVAL_SEC = 2.0
DISPLAY_WINDOW = True

# Congestion scoring normalization baseline
SATURATION_PER_LANE = 20

# Helmet violation module defaults
HELMET_MODEL_PATH = "helmet_detector.pt"
PLATE_MODEL_PATH = None
HELMET_BIKE_CLASS_ID = 3
HELMET_BIKE_CONF = 0.20
HELMET_NESTED_CONF = 0.35
HELMET_NO_HELMET_MIN_CONF = 0.45
HELMET_FINES_COOLDOWN_SEC = 5.0

