#!/bin/bash
set -e

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEMO_DIR=$(realpath "${SCRIPT_DIR}/..")
cd "$DEMO_DIR"

MODEL_NAME="yolo26n-seg.dxnn"
MODEL_DIR="assets/models"
MODEL_FILE="${MODEL_DIR}/${MODEL_NAME}"

# Prefer the model already shipped in the local DX resource tree; fall back to
# the public SDK download if it is not present on this machine.
LOCAL_MODEL="/home/radxa/workspace/workspace/res/models/models-2_3_0/${MODEL_NAME}"
MODEL_URL="https://sdk.deepx.ai/res/assets/dx_demo/${MODEL_NAME}"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "[INFO] Model already exists: $MODEL_FILE"
    exit 0
fi

if [ -f "$LOCAL_MODEL" ]; then
    echo "[INFO] Linking local model: $LOCAL_MODEL"
    ln -sf "$LOCAL_MODEL" "$MODEL_FILE"
    echo "[INFO] Model ready: $MODEL_FILE"
    exit 0
fi

echo "[INFO] Downloading $MODEL_FILE ..."
curl -f -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL" || {
    echo "[ERROR] Failed to download model from $MODEL_URL"
    echo "[ERROR] (local model not found at $LOCAL_MODEL either)"
    rm -f "$MODEL_FILE"
    exit 1
}

echo "[INFO] Model downloaded: $MODEL_FILE"
