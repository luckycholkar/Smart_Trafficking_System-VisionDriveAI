# Violation modules

All three violation pipelines live under `src/visiondrive/violations/` and
are toggled via `features.*` flags in `config.yaml`.

## Red-light violation

Always on. Module: `violations/red_light.py`.

**How it works.** Tracks per-vehicle `(track_id, last_y)`. When a vehicle's
center y crosses `stop_line_y` going downward AND the lane's signal is RED,
it emits one violation event:

```json
{ "frame": 832, "object_id": 14, "lane": "lane_2",
  "event": "STOP_LINE_CROSSED_ON_RED" }
```

Each `object_id` is flagged at most once per session.

**Tunables.** `detection.stop_line_ratio_y` (fraction of frame height).

## Helmet violation

Off by default. Enable with `features.enable_helmet: true`. Module:
`violations/helmet.py`.

**Pipeline.**
1. Vehicle detector finds two-wheelers (`bike_class_id`, default 3 = motorcycle).
2. Helmet detector runs on each two-wheeler crop.
3. If the crop has both a rider class AND a no-helmet class above
   `no_helmet_min_conf`, fire.
4. Plate ROI is extracted (via plate detector if configured, else
   lower-center fallback) and passed to the ANPR callable.
5. A fine record is appended; cooldown via IoU prevents duplicate fines for
   the same bike across consecutive frames.

**Required weights.** `data/weights/helmet_detector.pt` (path configurable
under `helmet.weights_path`). Not provided in this repo.

**Plug in real ANPR.** Replace `visiondrive.violations._anpr.run_anpr` with
an EasyOCR or vendor plate-recognition function and pass it to
`HelmetViolationPipeline(anpr_fn=...)`.

## Accident detection

Off by default. Enable with `features.enable_accident: true`. Module:
`violations/accident.py`.

**Heuristics.** No model. Operates on tracked detections from the main
pipeline. Two confirmation signals:

1. **Sudden impact**: vehicle goes from `>= high_speed_px_per_frame` to
   `<= near_zero_speed_px_per_frame` within `speed_drop_window_frames`.
2. **Collision + stationary**: two vehicles overlap (IoU >= threshold), the
   merged region barely moves for `stationary_seconds`.

**Side effects.** When enabled together with `features.enable_sms`, each
event is forwarded to `EmergencyAlert.notify_async()` and a JPG snapshot of
the frame is written under `accident.incidents_dir`.

**Severity scoring.** Combines the speed drop and collision IoU into HIGH /
MEDIUM / LOW for the SMS body.

## Twilio SMS

Off by default. Enable with `features.enable_sms: true` and set Twilio env
vars (see `.env.example`). Module: `notifications/twilio_sms.py`.

- Non-blocking: `notify_async()` queues, a worker thread sends.
- Cooldown: same incident-key (camera + trigger + involved vehicle ids)
  won't be re-sent within `notifications.cooldown_seconds`.
- Audit log: every queued event lands in `emergency_logs.csv` with status
  `SUCCESS` or `FAIL` plus a Twilio sid or error detail.
