from visiondrive.constants import EVENT_RED_LIGHT
from visiondrive.violations.red_light import RedLightViolationDetector


def test_no_violation_on_first_observation():
    det = RedLightViolationDetector(stop_line_y=100)
    out = det.update(object_id=1, center_y=200, lane_id="lane_1",
                     signal_map={"lane_1": "RED"}, frame_index=1)
    assert out is None


def test_flags_when_crossing_on_red():
    det = RedLightViolationDetector(stop_line_y=100)
    det.update(1, center_y=80, lane_id="lane_1", signal_map={"lane_1": "RED"}, frame_index=1)
    out = det.update(1, center_y=120, lane_id="lane_1", signal_map={"lane_1": "RED"}, frame_index=2)
    assert out is not None
    assert out["event"] == EVENT_RED_LIGHT
    assert out["object_id"] == 1


def test_does_not_flag_on_green():
    det = RedLightViolationDetector(stop_line_y=100)
    det.update(1, center_y=80, lane_id="lane_1", signal_map={"lane_1": "GREEN"}, frame_index=1)
    out = det.update(1, center_y=120, lane_id="lane_1", signal_map={"lane_1": "GREEN"}, frame_index=2)
    assert out is None


def test_flags_only_once_per_object():
    det = RedLightViolationDetector(stop_line_y=100)
    det.update(1, 80, "lane_1", {"lane_1": "RED"}, 1)
    first = det.update(1, 120, "lane_1", {"lane_1": "RED"}, 2)
    second = det.update(1, 140, "lane_1", {"lane_1": "RED"}, 3)
    assert first is not None
    assert second is None
