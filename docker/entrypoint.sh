#!/usr/bin/env bash
set -euo pipefail

# Auto-fetch default weights if the mount is empty (skipped if the user
# pre-provisioned the volume).
if [[ ! -f "data/weights/yolov8n.pt" ]]; then
  echo "[entrypoint] No default weights found, fetching yolov8n.pt..."
  bash scripts/download_weights.sh yolov8n || echo "[entrypoint] WARNING: weight download failed; pipeline may not start."
fi

exec "$@"
