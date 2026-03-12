#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

# Install missing Python dependencies from requirements.txt
if pip install --dry-run -r requirements.txt 2>/dev/null | grep -q "Would install"; then
    echo "[INFO] Installing missing Python dependencies..."
    pip install -r requirements.txt
fi

# Install dx-postprocess-seg C++ extension if not present
if ! pip list 2>/dev/null | grep -qi "dx-postprocess-seg"; then
    echo "[INFO] dx-postprocess-seg not found. Installing..."
    pip install src/bindings/python/dx_postprocess/
fi

# Download sample videos if not present
if [ ! "$(ls -A assets/videos/ 2>/dev/null)" ]; then
    echo "[INFO] Sample videos not found. Running setup.sh..."
    ./setup.sh
fi

echo "[INFO] Starting demo..."
python -m demo.main "$@"
