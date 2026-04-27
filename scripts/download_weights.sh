#!/usr/bin/env bash
# Fetch model weights for VisionDrive AI.
# Usage:
#   bash scripts/download_weights.sh                 # fetch defaults (yolov8n)
#   bash scripts/download_weights.sh yolov8s yolov8m # fetch additional Ultralytics checkpoints
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/data/weights"
mkdir -p "${WEIGHTS_DIR}"

DEFAULT_MODELS=(yolov8n)
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0"

for name in "${MODELS[@]}"; do
  out="${WEIGHTS_DIR}/${name}.pt"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${name}.pt already present at ${out}"
    continue
  fi
  echo "[fetch] ${name}.pt -> ${out}"
  curl -fL --progress-bar -o "${out}" "${BASE_URL}/${name}.pt"
done

echo "Done. Weights available under ${WEIGHTS_DIR}"
