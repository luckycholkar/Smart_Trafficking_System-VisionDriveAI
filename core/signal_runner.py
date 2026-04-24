from __future__ import annotations

import threading
import time
from typing import Dict

from config.settings import LANES, SIGNAL_RECOMPUTE_INTERVAL_SEC
from core.signal_logic import AdaptiveSignalController
from services.data_bus import DataBus


class SignalRunner:
    """
    Software-only traffic signal simulator.
    """

    def __init__(self, data_bus: DataBus):
        self.data_bus = data_bus
        self.controller = AdaptiveSignalController()
        self.stop_event = threading.Event()

    def run(self) -> None:
        current_green = LANES[0]
        until = time.time() + 10.0

        while not self.stop_event.is_set():
            snapshot = self.data_bus.snapshot()
            lane_counts: Dict[str, int] = snapshot.get("lane_counts") or {lane: 0 for lane in LANES}
            timings = self.controller.compute_timings(lane_counts)

            if time.time() >= until and timings:
                current_green, green_time = self.controller.pick_priority_lane(lane_counts)
                until = time.time() + green_time

            signal_state = {lane: ("GREEN" if lane == current_green else "RED") for lane in LANES}
            self.data_bus.update(
                signal_state=signal_state,
                signal_timings=timings,
                active_green_lane=current_green,
                green_remaining=max(0.0, round(until - time.time(), 2)),
            )
            time.sleep(SIGNAL_RECOMPUTE_INTERVAL_SEC)

