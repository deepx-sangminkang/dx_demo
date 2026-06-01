#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
pushd "$SCRIPT_DIR" > /dev/null

# Make the dx_stream GStreamer plugin discoverable for the dxstream backend,
# if install_dxstream.sh has been run (no-op for the legacy backend).
if [ -f "scripts/.dxstream_env.sh" ]; then
    # shellcheck disable=SC1091
    source "scripts/.dxstream_env.sh"
fi

# Install missing Python dependencies from requirements.txt
if pip install --dry-run -r requirements.txt 2>/dev/null | grep -q "Would install"; then
    echo "[INFO] Installing missing Python dependencies..."
    pip install -r requirements.txt
fi

# Download model and sample videos if not present
if [ ! -f "assets/models/yolo26n-1.dxnn" ] || [ ! "$(ls -A assets/videos/ 2>/dev/null)" ]; then
    echo "[INFO] Model or sample videos not found. Running setup.sh..."
    ./setup.sh
fi

echo "[INFO] Starting demo..."
python -m demo.main "$@"
popd > /dev/null
