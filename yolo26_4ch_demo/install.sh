#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

WITH_DXSTREAM=0
DXSTREAM_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --with-dxstream) WITH_DXSTREAM=1 ;;
        --dxstream-skip-deps) DXSTREAM_ARGS+=("--skip-deps") ;;
        --dxstream-prefix=*) DXSTREAM_ARGS+=("--prefix=${arg#*=}") ;;
        --help|-h)
            echo "Usage: ./install.sh [--with-dxstream] [--dxstream-skip-deps] [--dxstream-prefix=PATH]"
            echo ""
            echo "  --with-dxstream         Also build & install dx_stream (GStreamer plugins"
            echo "                          + pydxs) required by the dxstream inference backend."
            echo "  --dxstream-skip-deps    Pass --skip-deps to the dx_stream installer."
            echo "  --dxstream-prefix=PATH  Install dx_stream to PATH (default /usr/local)."
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
    echo "[INFO] Installing dx_stream (dxstream backend requirements)..."
    ./scripts/install_dxstream.sh "${DXSTREAM_ARGS[@]}"
else
    echo "[INFO] Skipping dx_stream install (legacy backend only)."
    echo "[HINT] The dxstream backend (engine_backend: dxstream) requires dx_stream."
    echo "[HINT] Re-run with: ./install.sh --with-dxstream  (or ./scripts/install_dxstream.sh)"
fi

echo "[INFO] Installation complete."
