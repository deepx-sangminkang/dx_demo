#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt

echo "[INFO] Enforcing a GStreamer-enabled OpenCV (required for HW decoding)..."
./scripts/ensure_gstreamer_opencv.sh

echo "[INFO] Downloading sample videos..."
./setup.sh

echo "[INFO] Installation complete."
