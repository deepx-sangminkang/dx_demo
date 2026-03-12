#!/bin/bash
set -e

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEMO_DIR=$(realpath "${SCRIPT_DIR}/..")
cd "$DEMO_DIR"

MODEL_URL="https://sdk.deepx.ai/res/assets/dx_demo/yolo11s-seg_optim.dxnn"
MODEL_DIR="assets/models"
MODEL_FILE="${MODEL_DIR}/yolo11s-seg_optim.dxnn"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "[INFO] Model already exists: $MODEL_FILE"
    exit 0
fi

echo "[INFO] Downloading $MODEL_FILE ..."
curl -f -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL" || {
    echo "[ERROR] Failed to download model from $MODEL_URL"
    rm -f "$MODEL_FILE"
    exit 1
}

echo "[INFO] Model downloaded: $MODEL_FILE"
