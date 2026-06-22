#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
pushd "$SCRIPT_DIR" > /dev/null

RUNTIME_DIR="${DX_RUNTIME_DIR:-$(dirname "$SCRIPT_DIR")/dx-runtime}"
DEMO_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-dir=*) RUNTIME_DIR="${1#*=}"; shift ;;
        --runtime-dir)   RUNTIME_DIR="$2"; shift 2 ;;
        *) DEMO_ARGS+=("$1"); shift ;;
    esac
done

# Activate dx_stream venv (installed by install.sh --target=dx_stream)
VENV="${RUNTIME_DIR}/dx_stream/venv-dx_stream/bin/activate"
if [ -f "$VENV" ]; then
    echo "[INFO] Using venv: ${VENV}"
    # shellcheck disable=SC1090
    source "$VENV"
else
    echo "[WARN] venv not found: ${VENV}"
    echo "[WARN] Run ./install.sh first to install dx_stream."
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
python -m demo.main "${DEMO_ARGS[@]}"
popd > /dev/null
