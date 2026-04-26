# VisionDrive AI

End-to-end Smart City system that turns standard CCTV feeds into adaptive
traffic-signal control, violation alerts, and emergency dispatch. Built
around an Ultralytics YOLO vision pipeline with a pluggable model
abstraction so new frameworks can be dropped in without touching the
pipeline.

## Features

- **Adaptive signal control** - per-lane vehicle density drives dynamic
  green/yellow/red timings via `AdaptiveSignalController`.
- **Red-light violation detection** - tracked-vehicle stop-line crossings
  on RED.
- **Helmet + ANPR pipeline** *(opt-in)* - two-stage nested detection,
  fines issued with cooldown.
- **Heuristic accident detection** *(opt-in)* - speed-drop and
  overlap+stationary triggers; Twilio SMS dispatch.
- **Live dashboard** - FastAPI + uvicorn, JSON state at `/api/state`.
- **Pluggable detectors** - `Detector` ABC + registry; today
  Ultralytics, tomorrow whatever you register.

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
make install-dev

# 2. Get default weights (yolov8n.pt) and a sample video
make weights
# (place your own traffic.mp4 under data/videos/, or update video.path in config)

# 3. Configure
cp .env.example .env

# 4. Run
make run
# Dashboard: http://127.0.0.1:5000
```

For just the vision pipeline (no API/dashboard): `make pipeline`.

## Repository layout

```
config/             yaml config (defaults + dev/prod overlays)
src/visiondrive/    package source - see docs/architecture.md
tests/              unit + integration tests
docs/               architecture, configuration, models, violations, deployment
docker/             container entrypoint
scripts/            weight downloader, dev launcher
data/               videos + weights (gitignored - use scripts/download_weights.sh)
```

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/architecture.md](docs/architecture.md) | Threading model, data flow, components |
| [docs/configuration.md](docs/configuration.md) | Every config key, env-var overrides |
| [docs/models.md](docs/models.md) | Adding a new model framework via the registry |
| [docs/violations.md](docs/violations.md) | Red-light, helmet, accident pipelines |
| [docs/deployment.md](docs/deployment.md) | Local, Docker, edge devices (Jetson, RPi) |
| [docs/development.md](docs/development.md) | Dev workflows, tests, code style |

## Tech stack

Python 3.10+, FastAPI, uvicorn, Ultralytics YOLO, OpenCV, PyTorch,
Pydantic v2 + pydantic-settings, Twilio (optional), EasyOCR (optional).

## License

Proprietary.
