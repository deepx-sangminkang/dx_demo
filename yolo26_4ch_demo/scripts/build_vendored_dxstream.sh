#!/bin/bash
# Build & install the vendored dx_stream subset (GStreamer plugin + pydxs) that
# the dxstream inference backend needs -- WITHOUT requiring a separate dx_stream
# checkout. The source lives in third_party/dxstream/ inside this demo.
#
# Only the elements the demo uses are built: dxpreprocess / dxinfer /
# dxpostprocess / dxscale (+ the metadata the pydxs bindings read).
#
# Usage:
#   scripts/build_vendored_dxstream.sh [--prefix=PATH] [--clean] [--force] [--skip-deps]
#
#   --prefix=PATH  Install prefix for the plugin (default:
#                  <demo>/third_party/dxstream/install -- local, no sudo).
#   --clean        Remove previous build/install artifacts first.
#   --force        Rebuild even if the plugin + pydxs already look installed.
#   --skip-deps    Skip the best-effort apt install of build dependencies.

set -u

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEMO_DIR=$(realpath -s "${SCRIPT_DIR}/..")
VENDOR_DIR="${DEMO_DIR}/third_party/dxstream"
PLUGIN_SRC="${VENDOR_DIR}/gst-dxstream-plugin"
PYDXS_SRC="${VENDOR_DIR}/bindings/python/pydxs"

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

PREFIX="${VENDOR_DIR}/install"
CLEAN=0
FORCE=0
SKIP_DEPS=0
for arg in "$@"; do
    case "$arg" in
        --prefix=*)  PREFIX="${arg#*=}" ;;
        --clean)     CLEAN=1 ;;
        --force)     FORCE=1 ;;
        --skip-deps) SKIP_DEPS=1 ;;
        --help|-h)   sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown option: $arg"; exit 1 ;;
    esac
done
PREFIX="$(realpath -m "${PREFIX}")"

PYBIN="${PYTHON:-}"
if [ -z "${PYBIN}" ]; then
    if command -v python >/dev/null 2>&1; then PYBIN="python"; else PYBIN="python3"; fi
fi

# --- sanity ------------------------------------------------------------------
if [ ! -f "${PLUGIN_SRC}/meson.build" ] || [ ! -f "${PYDXS_SRC}/setup.py" ]; then
    err "Vendored dx_stream source not found under ${VENDOR_DIR}."
    hint "Expected ${PLUGIN_SRC}/meson.build and ${PYDXS_SRC}/setup.py."
    exit 1
fi

pydxs_importable()      { "${PYBIN}" -c "import pydxs" >/dev/null 2>&1; }
plugin_env_file()       { echo "${SCRIPT_DIR}/.dxstream_env.sh"; }

dxstream_plugin_available() {
    "${PYBIN}" - <<'PYEOF' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
if not Gst.is_initialized():
    Gst.init(None)
import sys
sys.exit(0 if Gst.ElementFactory.find("dxpreprocess") else 1)
PYEOF
}

# Source any previously-written env so the availability check sees a local build.
if [ -f "$(plugin_env_file)" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$(plugin_env_file)"
    set -u
fi

if [ "$FORCE" -ne 1 ] && pydxs_importable && dxstream_plugin_available; then
    ok "Vendored dx_stream plugin + pydxs already available. Nothing to do."
    hint "Re-run with --force to rebuild."
    exit 0
fi

# --- optional build dependencies (best effort) -------------------------------
if [ "$SKIP_DEPS" -ne 1 ]; then
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing build dependencies (apt; best effort, sudo)..."
        SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
        ${SUDO} apt-get update -qq || warn "apt-get update failed; continuing."
        ${SUDO} apt-get install -y --no-install-recommends \
            meson ninja-build pkg-config build-essential \
            libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
            libjson-glib-dev libopencv-dev zlib1g-dev libdrm-dev \
            >/dev/null 2>&1 || warn "Some apt packages could not be installed; continuing."
    else
        warn "apt-get not found; skipping system-dependency install."
    fi
    "${PYBIN}" -m pip install --quiet "pybind11>=2.6.0" || warn "pybind11 pip install failed; continuing."
else
    log "Skipping system-dependency install (--skip-deps)."
fi

# --- clean -------------------------------------------------------------------
if [ "$CLEAN" -eq 1 ]; then
    log "Cleaning previous build/install artifacts..."
    rm -rf "${PLUGIN_SRC}/builddir" "${PREFIX}"
    rm -rf "${PYDXS_SRC}/build" "${PYDXS_SRC}/dist" "${PYDXS_SRC}"/*.egg-info
fi

# --- build & install the plugin ----------------------------------------------
log "Building vendored dxstream plugin (meson) -> ${PREFIX}..."
if [ ! -d "${PLUGIN_SRC}/builddir" ]; then
    if ! meson setup "${PLUGIN_SRC}/builddir" "${PLUGIN_SRC}" \
            --prefix="${PREFIX}" --buildtype=release; then
        err "meson setup failed."; exit 1
    fi
fi
if ! meson compile -C "${PLUGIN_SRC}/builddir"; then
    err "meson compile failed."; exit 1
fi
if ! meson install -C "${PLUGIN_SRC}/builddir"; then
    err "meson install failed."; exit 1
fi

PLUGIN_SO="$(find "${PREFIX}/lib" -name 'libgstdxstream.so' -path '*/gstreamer-1.0/*' 2>/dev/null | head -n1)"
if [ -z "${PLUGIN_SO}" ]; then
    err "Build finished but libgstdxstream.so was not found under ${PREFIX}/lib."
    exit 1
fi
PLUGIN_DIR="$(dirname "${PLUGIN_SO}")"
ok "Plugin installed: ${PLUGIN_SO}"

# --- write the env file (sourced by run_demo.sh) -----------------------------
ENV_FILE="$(plugin_env_file)"
cat > "${ENV_FILE}" <<EOF
# Auto-generated by build_vendored_dxstream.sh. Sourced by run_demo.sh so the
# dxstream backend can locate the vendored GStreamer plugin without a system
# install or editing ~/.bashrc.
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export GST_PLUGIN_PATH="${PLUGIN_DIR}:\${GST_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="${PLUGIN_DIR}:\${LD_LIBRARY_PATH:-}"
EOF
ok "Wrote environment file: ${ENV_FILE}"

# --- build & install pydxs ---------------------------------------------------
log "Building & installing pydxs bindings..."
set +u
# shellcheck disable=SC1090
source "${ENV_FILE}"
set -u
export PROJECT_ROOT="${VENDOR_DIR}"
if ! "${PYBIN}" -m pip install "${PYDXS_SRC}"; then
    err "pydxs pip install failed."
    exit 1
fi
ok "pydxs installed."

# --- verify ------------------------------------------------------------------
if pydxs_importable && dxstream_plugin_available; then
    ok "Vendored dx_stream build verified (plugin + pydxs available)."
    exit 0
fi
warn "Build finished but verification did not pass in this shell."
hint "Open a new shell and re-run the demo; run_demo.sh sources ${ENV_FILE}."
exit 0
