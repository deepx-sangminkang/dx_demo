#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

SKIP_DXSTREAM=0
DXSTREAM_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-dxstream) SKIP_DXSTREAM=1 ;;
        --with-dxstream) ;;  # accepted for backward compat; dxstream is now default
        --dxstream-runtime-dir=*) DXSTREAM_ARGS+=("--runtime-dir=${arg#*=}") ;;
        --help|-h)
            echo "Usage: ./install.sh [--skip-dxstream] [--dxstream-runtime-dir=PATH]"
            echo ""
            echo "  This demo is dx_stream-only (no OpenCV). dx_stream + pydxs are"
            echo "  installed by default via the dx-runtime installer."
            echo ""
            echo "  --skip-dxstream              Do not install dx_stream (assume it is"
            echo "                               already present on this machine)."
            echo "  --dxstream-runtime-dir=PATH  Path to a dx-runtime checkout."
            exit 0
            ;;
        *) echo "[WARN] Unknown option: $arg" ;;
    esac
done

echo "[INFO] Installing Python dependencies..."
pip install -r requirements.txt

echo "[INFO] Downloading sample videos..."
./setup.sh

if [ "$SKIP_DXSTREAM" -eq 0 ]; then
    echo "[INFO] Installing dxstream backend (dx_stream plugin + pydxs) via dx-runtime..."
    ./scripts/install_dxstream.sh "${DXSTREAM_ARGS[@]}"
else
    echo "[INFO] Skipping dxstream install (--skip-dxstream)."
    echo "[HINT] The demo REQUIRES the dx_stream plugin + pydxs to run."
    echo "[HINT] Verify with: gst-inspect-1.0 dxinfer && python -c 'import pydxs'"
fi

echo "[INFO] Installation complete."
