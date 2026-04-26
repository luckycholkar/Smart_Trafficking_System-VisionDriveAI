import pytest

from visiondrive.settings import SignalConfig
from visiondrive.signal.controller import AdaptiveSignalController


def make_controller() -> AdaptiveSignalController:
    return AdaptiveSignalController(SignalConfig(), saturation_per_lane=20)


def test_empty_lane_counts_returns_empty_timings():
    assert make_controller().compute_timings({}) == {}


def test_each_lane_gets_green_yellow_red():
    timings = make_controller().compute_timings({"a": 5, "b": 0, "c": 12})
    assert set(timings.keys()) == {"a", "b", "c"}
    for t in timings.values():
        assert {"green", "yellow", "red"} <= t.keys()


def test_busiest_lane_wins_priority():
    counts = {"a": 1, "b": 18, "c": 4}
    lane, green = make_controller().pick_priority_lane(counts)
    assert lane == "b"
    assert green > 0


def test_priority_lane_raises_on_empty():
    with pytest.raises(ValueError):
        make_controller().pick_priority_lane({})


def test_green_clamped_to_signal_bounds():
    cfg = SignalConfig(min_green=5.0, max_green=20.0)
    ctrl = AdaptiveSignalController(cfg, saturation_per_lane=20)
    timings = ctrl.compute_timings({"a": 100, "b": 0, "c": 0})
    for t in timings.values():
        assert cfg.min_green <= t["green"] <= cfg.max_green
