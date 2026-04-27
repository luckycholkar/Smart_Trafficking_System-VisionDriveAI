# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

This is a Python package under `src/visiondrive/` (src-layout, editable
install via `pip install -e .`). Configuration lives in `config/*.yaml`
plus `.env`; documentation lives in `docs/`.

```
src/visiondrive/
  settings.py         layered config loader (yaml + env, Pydantic v2)
  constants.py        immutable schema constants (COCO ids, DataBus keys)
  cli.py              `visiondrive run | pipeline | version`
  api/                FastAPI app, routes, schemas, dashboard.html
  core/               VisionPipeline, DataBus, drawing helpers
  signal/             AdaptiveSignalController, SignalRunner
  models/             Detector ABC + registry + UltralyticsDetector
  violations/         red_light (always on), helmet, accident
  notifications/      Twilio SMS dispatcher
```

## Common commands

All runnable via `make`:

```bash
make install-dev   # editable install + dev deps
make weights       # fetch yolov8n.pt into data/weights/
make run           # uvicorn + pipeline (http://127.0.0.1:5000)
make pipeline      # vision pipeline only, no API
make test          # pytest
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
make docker-build && make docker-run
```

A single test: `pytest tests/unit/test_signal_logic.py::test_busiest_lane_wins_priority`.

## Configuration

Resolution order (later wins): `config/config.yaml` -> `config/config.{env}.yaml`
(env from `VISIONDRIVE_ENV`, defaults to `dev`) -> `.env` / process env
prefixed with `VISIONDRIVE_` and nested via `__`.

Examples:
- `VISIONDRIVE_API__PORT=8080`
- `VISIONDRIVE_FEATURES__ENABLE_SMS=true`

`constants.py` is for non-tunable schema constants. Anything tunable goes
in `config.yaml` and is reflected in `settings.py` Pydantic models.

## Architecture (high-signal)

- **DataBus** (`core/data_bus.py`) is the contract between threads. Always
  read/write through it; never call across threads directly. Use the
  `BUS_*` constants from `constants.py` for keys.
- **VisionPipeline** (`core/pipeline.py`) spawns its own reader + inference
  threads and depends only on the `Detector` ABC (`models/base.py`).
  Swapping model frameworks is a config + registration change.
- **SignalRunner** (`signal/runner.py`) ticks every
  `api.recompute_interval_sec` (default 2 s), reads `lane_counts`, writes
  `signal_state` / `signal_timings` / `active_green_lane`.
- **FastAPI lifespan** (`api/app.py`) is what wires it all up: at startup
  it constructs the DataBus, builds the detector, and spawns pipeline +
  signal threads. At shutdown it sets stop events and joins.

The detector registry (`models/registry.py`) is the extension point for
new model frameworks - one file with a `register_detector("name", Cls)`
call is enough. See `docs/models.md`.

## Feature flags

Helmet, accident, and SMS are off by default. Toggle in
`config.{env}.yaml` or via `VISIONDRIVE_FEATURES__*`. They each require
extra setup (helmet needs `data/weights/helmet_detector.pt`; SMS needs
Twilio env vars).

## Testing without weights

`tests/conftest.py` provides `FakeDetector` that satisfies the `Detector`
interface with scripted detections. Use it for unit tests so they run
without ultralytics or `.pt` files on disk - that's the whole reason the
abstraction exists.

## Things that will burn you

- The vision pipeline `display_window` flag uses `cv2.imshow`. In Docker
  / SSH / CI this must be `false` (it's already false in
  `config.prod.yaml`).
- `data/` is gitignored. Weights and videos are mounted in at runtime in
  Docker; locally use `make weights` to fetch them.
- The legacy stdlib HTTP dashboard at `dashboard/dashboard.py` is gone -
  it's now FastAPI under `src/visiondrive/api/`. Do not re-introduce
  threading-server code.
- `app.py`, `Prototype.py`, and the old `core/` / `services/` / `stubs/`
  directories are deleted. Use `src/visiondrive/` and the
  `examples/prototype_demo.py` script.
