# Configuration

VisionDrive uses three layers, merged in this order (later wins):

1. **`config/config.yaml`** - committed defaults.
2. **`config/config.{env}.yaml`** - per-environment overrides, where `env` is
   the value of the `VISIONDRIVE_ENV` env var (defaults to `dev`).
3. **`.env` + process environment** - prefixed `VISIONDRIVE_`, nested via `__`.

The result is a `Settings` Pydantic model. Access it from code via
`from visiondrive.settings import get_settings`.

## Why three layers

| Layer | Purpose | Example |
|-------|---------|---------|
| `config.yaml` | Tunable knobs that should be reviewed in PRs | detection thresholds, lane definitions, signal timing |
| `config.{env}.yaml` | Differences between dev and prod that aren't secret | `display_window: false` in prod |
| `.env` / env vars | Per-deployment values + secrets | Twilio creds, dashboard port for a specific host |

`constants.py` holds values that aren't tunable - data-schema facts like
COCO class IDs or DataBus key names.

## Overriding via env vars

Nested keys use `__` as a separator:

```bash
VISIONDRIVE_API__PORT=8080
VISIONDRIVE_FEATURES__ENABLE_SMS=true
VISIONDRIVE_MODEL__WEIGHTS_PATH=/opt/models/yolov10s.pt
```

See `.env.example` for a starter template.

## Configuration keys

### `env`, `log_level`
Top-level. `env` selects which overlay file to apply. `log_level` is forwarded
to Python `logging` and uvicorn.

### `api`
| Key | Default | Notes |
|-----|---------|-------|
| `host` | `127.0.0.1` | Use `0.0.0.0` in containers. |
| `port` | `5000` | |
| `recompute_interval_sec` | `2.0` | How often the signal runner ticks. |
| `state_poll_interval_ms` | `1000` | How often the dashboard polls `/api/state`. |

### `video`
| Key | Default | Notes |
|-----|---------|-------|
| `path` | `data/videos/traffic.mp4` | Resolved relative to repo root if not absolute. |
| `output_path` | `data/videos/hackathon_demo_output.avi` | Annotated output written here. |
| `output_fps` | `25.0` | |

### `model`
| Key | Default | Notes |
|-----|---------|-------|
| `framework` | `ultralytics` | Must match a name registered via `register_detector`. |
| `weights_path` | `data/weights/yolov8n.pt` | Path to the checkpoint. |
| `tracker_config` | `bytetrack.yaml` | Tracker preset (Ultralytics built-in). |
| `conf` | `0.15` | Detection confidence threshold. |
| `iou` | `0.45` | NMS IoU threshold. |
| `imgsz` | `960` | Inference resolution. |

### `detection`
| Key | Default | Notes |
|-----|---------|-------|
| `vehicle_class_ids` | `[1,2,3,5,7]` | COCO: bicycle, car, motorcycle, bus, truck. |
| `saturation_per_lane` | `20` | Vehicles-per-lane that map to 100% pressure. |
| `stop_line_ratio_y` | `0.62` | Stop line position as a fraction of frame height. |

### `lanes`
| Key | Default | Notes |
|-----|---------|-------|
| `ids` | `["lane_1","lane_2","lane_3","lane_4"]` | Frame width is split equally across these lanes. |

### `signal`
Adaptive signal timing model. See `signal/controller.py` for the math.

### `features`
Feature toggles. Default-off ones require additional setup (weights, env vars).

| Key | Default | Notes |
|-----|---------|-------|
| `display_window` | `true` | OpenCV imshow window. Set false in headless / Docker. |
| `enable_helmet` | `false` | Requires `data/weights/helmet_detector.pt`. |
| `enable_accident` | `false` | Heuristic; needs no extra weights. |
| `enable_sms` | `false` | Requires Twilio env vars. |

### `helmet`, `accident`, `notifications`
Per-feature config, only used when the matching feature flag is true. See
`docs/violations.md` for runtime behavior details.
