import pytest

from visiondrive.models.registry import (
    detector_registry,
    get_detector_factory,
    register_detector,
)


def test_ultralytics_registered_via_import_side_effect():
    assert "ultralytics" in detector_registry


def test_unknown_framework_raises():
    with pytest.raises(KeyError, match="Unknown detector framework"):
        get_detector_factory("nonexistent")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_detector("ultralytics", lambda **_: None)
