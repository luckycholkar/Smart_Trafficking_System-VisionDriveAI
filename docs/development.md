# Development

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev          # installs runtime + dev extras
make weights              # default YOLO weights into data/weights/
cp .env.example .env
```

## Common workflows

```bash
make run            # uvicorn + pipeline (foreground)
make pipeline       # vision pipeline only (no API)
make test           # pytest
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
```

## Project layout

```
src/visiondrive/
  settings.py       layered yaml + env config (Pydantic)
  constants.py      immutable schema constants (COCO ids, bus keys)
  cli.py            `visiondrive run | pipeline | version`
  api/              FastAPI app, routes, schemas, dashboard.html template
  core/             VisionPipeline, DataBus, drawing helpers
  signal/           AdaptiveSignalController + SignalRunner
  models/           Detector ABC + registry + UltralyticsDetector
  violations/       red_light, helmet, accident
  notifications/    Twilio SMS dispatcher

tests/
  unit/             pure-logic tests (no GPU/weights required)
  integration/      API smoke tests using FastAPI TestClient
```

## Testing without weights

`tests/conftest.py` defines `FakeDetector` that satisfies the `Detector`
interface with scripted detections. Use it whenever you need to test the
pipeline or violation modules in isolation - it's why the abstraction
exists.

## Adding a new violation

1. New module under `src/visiondrive/violations/`.
2. Wire it into `core/pipeline.py` behind a feature flag in
   `settings.py` -> `FeaturesConfig`.
3. Add config block in `config.yaml`.
4. Tests in `tests/unit/` using `FakeDetector`.
5. Document in `docs/violations.md`.

## Adding a new model framework

See [models.md](models.md). Quick version:
1. New file in `src/visiondrive/models/`.
2. Subclass `Detector`, call `register_detector("name", Cls)` at module load.
3. Add an import to `src/visiondrive/models/__init__.py`.
4. Set `model.framework: name` in your config.

## Code style

- ruff (`make lint`, `make format`) - line length 110, select `E F I B UP SIM N RUF`.
- Type hints everywhere; mypy is set to non-strict to keep velocity high.
- Use the `BUS_*` constants when reading/writing `DataBus`. Stringly-typed
  keys are the #1 way producers and consumers drift.

## Release flow

1. Bump version in `pyproject.toml` and `src/visiondrive/__init__.py`.
2. `make test && make lint`.
3. Tag: `git tag vX.Y.Z && git push --tags`.
4. Build & push the image: `docker compose build && docker push ...`.
