#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

WITH_DXSTREAM=0
DXSTREAM_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --with-dxstream) WITH_DXSTREAM=1 ;;
        --dxstream-runtime-dir=*) DXSTREAM_ARGS+=("--runtime-dir=${arg#*=}") ;;
        --help|-h)
            echo "Usage: ./install.sh [--with-dxstream] [--dxstream-runtime-dir=PATH]"
            echo ""
            echo "  --with-dxstream            Also install the dx_stream GStreamer plugin"
            echo "                             + pydxs required by the dxstream backend, via"
            echo "                             the dx-runtime installer (--target=dx_stream)."
            echo "  --dxstream-runtime-dir=PATH  Path to a dx-runtime checkout."
            echo ""
            echo "Without --with-dxstream only the Python (legacy backend) demo is installed."
            exit 0
            ;;
        *) echo "[WARN] Unknown option: $arg" ;;
    esac
done

echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt

echo "[INFO] Enforcing a GStreamer-enabled OpenCV (required for HW decoding)..."
./scripts/ensure_gstreamer_opencv.sh

echo "[INFO] Downloading sample videos..."
./setup.sh

if [ "$WITH_DXSTREAM" -eq 1 ]; then
    echo "[INFO] Installing dxstream backend (dx_stream plugin + pydxs) via dx-runtime..."
    ./scripts/install_dxstream.sh "${DXSTREAM_ARGS[@]}"
else
    echo "[INFO] Skipping dxstream install (legacy backend only)."
    echo "[HINT] The dxstream backend (engine_backend: dxstream) needs the dx_stream plugin."
    echo "[HINT] Install it via dx-runtime: ./install.sh --with-dxstream"
    echo "[HINT]   (or, directly: <dx-runtime>/install.sh --target=dx_stream)"
fi

echo "[INFO] Installation complete."
