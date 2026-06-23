#!/bin/bash
set -e
SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/color_env.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/common_util.sh"

RUNTIME_REPO="https://github.com/DEEPX-AI/dx-runtime"
RUNTIME_DIR="${DX_RUNTIME_DIR:-$(dirname "$SCRIPT_DIR")/dx-runtime}"
SKIP_DXRT=0
SKIP_DXSTREAM=0
FORCE=0

show_help() {
    print_colored "Usage: ./install.sh [OPTIONS]" "YELLOW"
    print_colored "  --runtime-dir=PATH   Path to dx-runtime checkout (default: ../dx-runtime)" "GREEN"
    print_colored "  --skip-dxrt          Skip NPU driver/RT/FW installation" "GREEN"
    print_colored "  --skip-dxstream      Skip dxstream plugin + pydxs installation" "GREEN"
    print_colored "  -f, --force          Reinstall even if already present" "GREEN"
    print_colored "  -h, --help           Show this help" "GREEN"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --runtime-dir=*) RUNTIME_DIR="${arg#*=}" ;;
        --skip-dxrt)     SKIP_DXRT=1 ;;
        --skip-dxstream) SKIP_DXSTREAM=1 ;;
        # ponytail: backward compat aliases kept silent
        --with-dxstream|--dxstream-runtime-dir=*) ;;
        -f|--force)      FORCE=1 ;;
        -h|--help)       show_help ;;
        *) print_colored "Unknown option: $arg" "WARNING" ;;
    esac
done

ensure_cloned() {
    if [ -x "${RUNTIME_DIR}/install.sh" ]; then
        return 0
    fi
    print_colored "dx-runtime not found at ${RUNTIME_DIR}. Cloning..." "INFO"
    git clone --recurse-submodules "$RUNTIME_REPO" "$RUNTIME_DIR"
    ( cd "$RUNTIME_DIR" && git submodule update --init --recursive )
}

# Step 1: NPU driver / RT / FW
if [ "$SKIP_DXRT" -eq 0 ]; then
    if [ "$FORCE" -eq 1 ] || ! dxrt-cli -s >/dev/null 2>&1; then
        print_colored "dxrt-cli check failed. Installing NPU driver/RT/FW..." "INFO"
        ensure_cloned
        "$RUNTIME_DIR/install.sh" --target=dx_rt_npu_linux_driver
        "$RUNTIME_DIR/install.sh" --target=dx_rt
        "$RUNTIME_DIR/install.sh" --target=dx_fw
    else
        print_colored "dxrt-cli OK. Skipping NPU driver/RT/FW." "SKIP"
    fi
fi

# Step 2: dxstream (GStreamer plugin + pydxs)
if [ "$SKIP_DXSTREAM" -eq 0 ]; then
    DXSTREAM_NEEDED=0
    if [ "$FORCE" -eq 1 ]; then
        DXSTREAM_NEEDED=1
    elif ! gst-inspect-1.0 dxstream >/dev/null 2>&1; then
        DXSTREAM_NEEDED=1
    elif ! pip list 2>/dev/null | grep -q pydxs; then
        DXSTREAM_NEEDED=1
    fi

    if [ "$DXSTREAM_NEEDED" -eq 1 ]; then
        print_colored "dxstream or pydxs not found. Installing via dx-runtime..." "INFO"
        ensure_cloned
        "$RUNTIME_DIR/install.sh" --target=dx_stream --sanity-check=n
        VENV="${RUNTIME_DIR}/dx_stream/venv-dx_stream/bin/activate"
        if [ -f "$VENV" ]; then
            # shellcheck disable=SC1090
            source "$VENV"
        fi
        if ! gst-inspect-1.0 dxstream >/dev/null 2>&1; then
            print_colored "dxstream install failed: gst-inspect-1.0 dxstream not found." "ERROR"
            exit 1
        fi
        print_colored "dxstream installed successfully." "OK"
    else
        print_colored "dxstream already installed." "SKIP"
    fi
fi

# Step 3: Python dependencies + assets
print_colored "Installing Python dependencies..." "INFO"
pip install -r requirements.txt

print_colored "Downloading model and sample videos..." "INFO"
./setup.sh

print_colored "Installation complete." "OK"
