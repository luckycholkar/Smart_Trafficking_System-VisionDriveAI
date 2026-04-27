# Architecture

## Component overview

```
                    +----------------------+
                    |   FastAPI lifespan   |
                    |  (visiondrive.api)   |
                    +----------+-----------+
                               |
              spawns           v          spawns
        +----------------+  shared  +----------------+
        | VisionPipeline |--------> |  SignalRunner  |
        | (reader+infer) |  DataBus | (every 2s)     |
        +----------------+   ^  ^   +----------------+
                |            |  |
                v            |  |
          Detector (ABC)     |  +-- updates signal_state
                |            |
                v            |
        UltralyticsDetector  |
        (visiondrive.models) |
                             |
   updates frame_index, lane_counts, congestion, fps,
   appends violations
```

## Threads

| Thread | Lives in | Job | Reads | Writes |
|--------|----------|-----|-------|--------|
| `vd-pipeline` reader | `core/pipeline.py` | OpenCV `VideoCapture` -> queue | Video file | Frame queue |
| `vd-pipeline` infer | `core/pipeline.py` | YOLO inference + lane assignment | Frame queue, `signal_state` | `lane_counts`, `vehicle_count`, `fps`, `frame_index`, violations |
| `vd-signal` | `signal/runner.py` | Recompute signal timings every `api.recompute_interval_sec` | `lane_counts` | `signal_state`, `signal_timings`, `active_green_lane`, `green_remaining` |
| `vd-emergency` | `notifications/twilio_sms.py` | Drain alert queue, dispatch SMS | Internal queue | CSV log |
| FastAPI worker(s) | `api/app.py` | Serve `/`, `/api/state`, `/health` | `DataBus` snapshot | nothing |

All cross-thread communication goes through the `DataBus` (`core/data_bus.py`).
It's a lock-protected dict; `snapshot()` returns a deep copy so consumers
can't mutate producer state.

## Why a Detector abstraction

Today the only implementation is `UltralyticsDetector`. Tomorrow you may want
a custom YOLO variant, RT-DETR, or a quantized ONNX runtime model. The
pipeline depends only on `Detector`, so swapping is a config change. See
[models.md](models.md).

## Lifecycle

1. `uvicorn visiondrive.api.app:app` -> `create_app()` returns a FastAPI instance.
2. FastAPI lifespan hook runs at startup:
   - reads merged settings (yaml + env),
   - constructs `DataBus`,
   - builds detector via `build_detector(settings.model)`,
   - spawns pipeline + signal threads (daemon=True).
3. Requests to `/api/state` snapshot the DataBus and serialize via Pydantic.
4. On shutdown, lifespan sets `stop_event`s and joins threads with timeouts.

## Where extensions plug in

| You want to... | Edit |
|----------------|------|
| Add a new model framework | New file in `src/visiondrive/models/` + `register_detector()` |
| Add a new violation type | New module in `src/visiondrive/violations/` + wire into pipeline |
| Replace the dashboard | New routes in `api/routes.py` (HTML + JSON live separately) |
| Replace SMS with email/webhook | New class alongside `notifications/twilio_sms.py` |
| Add a different signal strategy | Subclass or replace `signal/controller.py`, swap in `signal/runner.py` |
