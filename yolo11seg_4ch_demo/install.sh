#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt

echo "[INFO] Building and installing dx_postprocess_seg extension..."
pip install src/bindings/python/dx_postprocess/

echo "[INFO] Downloading sample videos..."
./setup.sh

echo "[INFO] Installation complete."
