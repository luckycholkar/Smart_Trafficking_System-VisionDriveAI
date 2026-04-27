from visiondrive.constants import BUS_FRAME_INDEX, BUS_VIOLATIONS
from visiondrive.core.data_bus import DataBus


def test_update_and_snapshot_roundtrip():
    bus = DataBus()
    bus.update(frame_index=42, vehicle_count=7)
    snap = bus.snapshot()
    assert snap[BUS_FRAME_INDEX] == 42
    assert snap["vehicle_count"] == 7


def test_snapshot_is_a_copy():
    bus = DataBus()
    bus.update(lane_counts={"lane_1": 1})
    snap = bus.snapshot()
    snap["lane_counts"]["lane_1"] = 999
    assert bus.snapshot()["lane_counts"]["lane_1"] == 1


def test_violations_capped_to_keep_window():
    bus = DataBus(violations_keep=3)
    for i in range(10):
        bus.append_violation({"i": i})
    assert [v["i"] for v in bus.snapshot()[BUS_VIOLATIONS]] == [7, 8, 9]
