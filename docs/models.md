# Models & detector abstraction

The pipeline talks to a `Detector` interface, not directly to Ultralytics or
any other framework. This keeps the door open for new model frameworks
without forking the pipeline.

## The contract

`src/visiondrive/models/base.py`:

```python
class Detector(ABC):
    @abstractmethod
    def detect(self, frame, *, classes=None) -> list[Detection]: ...

    @abstractmethod
    def track(self, frame, *, classes=None) -> list[TrackedDetection]: ...

    @property
    @abstractmethod
    def framework(self) -> str: ...
```

`Detection` and `TrackedDetection` are framework-agnostic dataclasses
holding a bbox, class id, confidence, and (for tracked) a track id.

## Adding a new framework

1. Create `src/visiondrive/models/<myframework>.py`.
2. Implement a class that subclasses `Detector`.
3. Call `register_detector("<name>", MyClass)` at module load.
4. Add the import to `src/visiondrive/models/__init__.py` so registration runs
   on package import.
5. Set `model.framework: <name>` in `config.yaml` (or via
   `VISIONDRIVE_MODEL__FRAMEWORK=<name>`).

Skeleton:

```python
# src/visiondrive/models/myrtdetr.py
from visiondrive.models.base import Detection, Detector, TrackedDetection
from visiondrive.models.registry import register_detector


class MyRTDETRDetector(Detector):
    framework_name = "rtdetr"

    def __init__(self, weights_path, *, conf=0.25, iou=0.45, imgsz=640, tracker_config="bytetrack.yaml"):
        # load your model here
        ...

    @property
    def framework(self) -> str:
        return self.framework_name

    def detect(self, frame, *, classes=None) -> list[Detection]:
        ...

    def track(self, frame, *, classes=None) -> list[TrackedDetection]:
        ...


register_detector("rtdetr", MyRTDETRDetector)
```

## Constructor signature

The factory in `models/factory.py` builds detectors by passing keyword args
from `ModelConfig`:

```python
factory(
    weights_path=...,
    conf=...,
    iou=...,
    imgsz=...,
    tracker_config=...,
)
```

If your framework needs additional knobs, extend `ModelConfig` in
`settings.py` and pass them through. Don't put framework-specific params on
the `Detector` interface itself.

## Helmet / plate detectors

The helmet violation pipeline takes its own `Detector` instances (one for
the vehicle pass, one for the nested helmet pass, optionally one for plate
ROI). They reuse the same registry, so any framework you add is immediately
usable for nested-detection pipelines too.

## Why not just import YOLO directly?

Three reasons:
- **Testability**: a `FakeDetector` in `tests/conftest.py` replaces the real
  one, so unit tests don't need weights on disk.
- **Plug-in surface**: research teams often want to A/B a custom model. The
  registry lets them ship one file instead of forking the pipeline.
- **Edge swaps**: production may want a quantized ONNX/TensorRT runtime
  while dev uses Ultralytics. Same code, different registration.
