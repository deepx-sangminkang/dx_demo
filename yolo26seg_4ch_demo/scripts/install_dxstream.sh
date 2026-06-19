#!/bin/bash
# Install the dx_stream GStreamer plugin (dxpreprocess / dxinfer / dxpostprocess
# / dxscale) and its pydxs Python bindings, which the native inference backend
# (engine_backend: dxstream) depends on.
#
# dx_stream is installed SEPARATELY from this demo, via the official DeepX
# dx-runtime installer:
#
#     <dx-runtime>/install.sh --target=dx_stream
#
# The legacy backend does NOT need this script. This helper just locates a
# dx-runtime checkout and runs that installer; if none is found it prints the
# clone-and-run instructions.
#
# Usage:
#   scripts/install_dxstream.sh [--runtime-dir=PATH] [--force]
#
#   --runtime-dir=PATH  Path to a dx-runtime checkout (default: $DX_RUNTIME_DIR,
#                       then common sibling locations).
#   --force             Run the installer even if dxstream already appears
#                       registered with GStreamer.

set -u

SCRIPT_DIR=$(realpath "$(dirname "$0")")

if [ -f "${SCRIPT_DIR}/color_env.sh" ] && [ -f "${SCRIPT_DIR}/common_util.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/color_env.sh"
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/common_util.sh"
else
    print_colored() { echo "[$2] $1"; }
fi

log()  { print_colored "$1" "INFO"; }
ok()   { print_colored "$1" "OK"; }
warn() { print_colored "$1" "WARNING"; }
err()  { print_colored "$1" "ERROR"; }
hint() { print_colored "$1" "HINT"; }

RUNTIME_DIR="${DX_RUNTIME_DIR:-}"
RUNTIME_REPO="${DX_RUNTIME_REPO:-https://github.com/DEEPX-AI/dx-runtime}"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --runtime-dir=*) RUNTIME_DIR="${arg#*=}" ;;
        --force)         FORCE=1 ;;
        --help|-h)
            sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) warn "Unknown option: $arg" ;;
    esac
done

dxstream_registered() {
    command -v gst-inspect-1.0 >/dev/null 2>&1 || return 1
    gst-inspect-1.0 dxinfer >/dev/null 2>&1
}

if [ "$FORCE" -eq 0 ] && dxstream_registered; then
    ok "dx_stream is already installed (dxinfer element is registered)."
    exit 0
fi

# Locate a dx-runtime checkout.
if [ -z "$RUNTIME_DIR" ]; then
    for candidate in \
        "$HOME/workspace/dx-runtime" \
        "$(realpath -s "${SCRIPT_DIR}/../../../dx-runtime" 2>/dev/null)" \
        "$(realpath -s "${SCRIPT_DIR}/../../dx-runtime" 2>/dev/null)"; do
        if [ -n "$candidate" ] && [ -x "$candidate/install.sh" ]; then
            RUNTIME_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$RUNTIME_DIR" ] || [ ! -x "$RUNTIME_DIR/install.sh" ]; then
    err "Could not find a dx-runtime checkout with install.sh."
    hint "Clone it and install the dx_stream target:"
    hint "    git clone --recurse-submodules ${RUNTIME_REPO}"
    hint "    cd dx-runtime && ./install.sh --target=dx_stream"
    hint "Then re-run this script, or set DX_RUNTIME_DIR=/path/to/dx-runtime."
    exit 1
fi

log "Installing dx_stream via ${RUNTIME_DIR}/install.sh --target=dx_stream ..."
( cd "$RUNTIME_DIR" && ./install.sh --target=dx_stream )
status=$?

if [ "$status" -ne 0 ]; then
    err "dx-runtime install.sh --target=dx_stream failed (exit ${status})."
    exit "$status"
fi

if dxstream_registered; then
    ok "dx_stream installed; dxinfer element is registered with GStreamer."
else
    warn "Installer finished but dxinfer is not yet visible to gst-inspect-1.0."
    hint "Ensure the dx_stream plugin dir is on GST_PLUGIN_PATH (it is usually"
    hint "installed under /usr/local/lib/<arch>/gstreamer-1.0)."
fi
