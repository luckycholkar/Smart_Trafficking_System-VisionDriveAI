import threading
import _thread as thread 

from core.signal_runner import SignalRunner
from core.vision_processing import VisionPipeline
from dashboard.dashboard import DashboardServer
from services.data_bus import DataBus


def main() -> None:
    data_bus = DataBus()
    dashboard = DashboardServer(data_bus)
    dashboard.run_in_thread()

    signal_runner = SignalRunner(data_bus)
    signal_thread = threading.Thread(target=signal_runner.run, daemon=True)
    signal_thread.start()

    pipeline = VisionPipeline(video_path="traffic.mp4", model_path="yolov8n.pt", data_bus=data_bus)
    try:
        pipeline.run()
    finally:
        signal_runner.stop_event.set()
        print("Demo complete. Output saved as hackathon_demo_output.avi")
        print("Dashboard was available at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()

