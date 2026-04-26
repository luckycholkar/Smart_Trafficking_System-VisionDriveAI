# Deployment

## Local (uvicorn)

```bash
make install-dev          # editable install + dev tools
make weights              # fetch yolov8n.pt into data/weights
cp .env.example .env      # adjust as needed
make run                  # uvicorn visiondrive.api.app:app
```

The dashboard is at `http://127.0.0.1:5000`. JSON state at `/api/state`,
health at `/health`.

`make pipeline` runs the vision pipeline standalone (no API/dashboard) -
useful when iterating on the detection code with `display_window: true`.

## Docker

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f
```

Volumes:
- `./data` -> `/app/data` (videos + weights stay outside the image).
- `./incidents` -> `/app/incidents` (accident snapshots).
- `./emergency_logs.csv` -> `/app/emergency_logs.csv` (Twilio audit log).

`Dockerfile` is multi-stage: builder installs deps into `/install`, runtime
copies them in. OpenCV's runtime needs `libgl1`, `libglib2.0-0`, and
`ffmpeg` - the runtime stage installs them.

The container runs `prod` config by default
(`config/config.prod.yaml`). Override anything via the `environment:` block
in `docker-compose.yml` or by editing `.env`.

### Healthcheck

The container exposes a healthcheck against `/health`. Use it from k8s
liveness/readiness probes or compose-level orchestration.

## Edge devices

### Jetson Nano / Orin

- Use the NVIDIA L4T base image instead of `python:3.11-slim`. The
  `Dockerfile` is a starting point; replace `FROM` lines and skip
  `pip install torch` (use the pre-installed JetPack PyTorch wheel).
- Enable hardware encoding: install `nvidia-container-toolkit` and add
  `runtime: nvidia` to the compose service.
- Set `model.imgsz` to 640 and consider switching `model.weights_path` to
  a quantized ONNX or TensorRT engine; register a corresponding
  `Detector` (see [models.md](models.md)).

### Raspberry Pi 4/5

- Stick to YOLOv8n at `imgsz: 480` for realistic FPS.
- Disable the OpenCV display: `features.display_window: false`.
- Use the `prod` overlay; set `model.framework` to a CPU-friendly backend
  if/when one is registered.

## Production checklist

- [ ] `VISIONDRIVE_ENV=prod`
- [ ] `features.display_window=false`
- [ ] `api.host=0.0.0.0` (already in `config.prod.yaml`)
- [ ] Twilio creds in `.env` if `enable_sms=true`
- [ ] Weights bind-mounted (don't bake into image)
- [ ] Persistent volumes for `incidents/` and `emergency_logs.csv`
- [ ] Reverse proxy (nginx/Caddy) terminating TLS in front of port 5000
- [ ] Container healthchecks wired into your orchestrator
