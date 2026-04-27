#!/usr/bin/env bash
# Local dev launcher: ensures weights exist, then starts uvicorn with reload.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "data/weights/yolov8n.pt" ]]; then
  bash scripts/download_weights.sh yolov8n
fi

export VISIONDRIVE_ENV="${VISIONDRIVE_ENV:-dev}"

exec uvicorn visiondrive.api.app:app \
  --host "${VISIONDRIVE_API__HOST:-127.0.0.1}" \
  --port "${VISIONDRIVE_API__PORT:-5000}" \
  --log-level info
