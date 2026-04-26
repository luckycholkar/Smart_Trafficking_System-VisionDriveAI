"""SignalRunner: drives the adaptive controller on a worker thread."""

from __future__ import annotations

import logging
import threading
import time

from visiondrive.constants import (
    BUS_ACTIVE_GREEN_LANE,
    BUS_GREEN_REMAINING,
    BUS_LANE_COUNTS,
    BUS_SIGNAL_STATE,
    BUS_SIGNAL_TIMINGS,
    SIGNAL_GREEN,
    SIGNAL_RED,
)
from visiondrive.core.data_bus import DataBus
from visiondrive.settings import Settings
from visiondrive.signal.controller import AdaptiveSignalController

log = logging.getLogger(__name__)


class SignalRunner:
    def __init__(self, settings: Settings, data_bus: DataBus) -> None:
        self._settings = settings
        self._data_bus = data_bus
        self._controller = AdaptiveSignalController(
            settings.signal, settings.detection.saturation_per_lane
        )
        self.stop_event = threading.Event()

    def run(self) -> None:
        lanes = self._settings.lanes.ids
        current_green = lanes[0]
        until = time.time() + 10.0
        interval = self._settings.api.recompute_interval_sec

        while not self.stop_event.is_set():
            snapshot = self._data_bus.snapshot()
            lane_counts = snapshot.get(BUS_LANE_COUNTS) or {lane: 0 for lane in lanes}
            timings = self._controller.compute_timings(lane_counts)

            if time.time() >= until and timings:
                current_green, green_time = self._controller.pick_priority_lane(lane_counts)
                until = time.time() + green_time

            signal_state = {
                lane: (SIGNAL_GREEN if lane == current_green else SIGNAL_RED) for lane in lanes
            }
            self._data_bus.update(
                **{
                    BUS_SIGNAL_STATE: signal_state,
                    BUS_SIGNAL_TIMINGS: timings,
                    BUS_ACTIVE_GREEN_LANE: current_green,
                    BUS_GREEN_REMAINING: max(0.0, round(until - time.time(), 2)),
                }
            )
            time.sleep(interval)

        log.info("Signal runner stopped")
