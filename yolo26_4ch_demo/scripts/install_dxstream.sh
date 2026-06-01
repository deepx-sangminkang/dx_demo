#!/bin/bash
# Build & install the dx_stream GStreamer plugins + pydxs bindings required by
# the native inference backend (engine_backend: dxstream).
#
# The legacy backend does NOT need this script. dx_stream is only required for
# the native GStreamer inference path (dxpreprocess / dxinfer / dxpostprocess)
# and its pydxs Python bindings. dx_stream itself depends on DX-RT (dx_engine)
# being installed first (same prerequisite the demo already documents).
#
# Behaviour:
#   * Idempotent  : if the dxstream plugin and pydxs are already importable it
#                   exits 0 without rebuilding.
#   * Self-locating: finds the dx_stream source via $DX_STREAM_SRC, common
#                   sibling paths, or clones it from $DX_STREAM_REPO.
#   * Graceful    : prints actionable guidance instead of cryptic failures when
#                   prerequisites (git, DX-RT) are missing.
#
# Usage:
#   scripts/install_dxstream.sh [--skip-deps] [--prefix=PATH] [--force]
#
#   --skip-deps   Skip dx_stream's system-dependency install.sh (apt/cmake/...).
#                 Use when the build toolchain is already present.
#   --prefix=PATH Installation prefix passed to dx_stream build.sh (default
#                 /usr/local). Plugins land in PREFIX/lib/<arch>/gstreamer-1.0.
#   --force       Reinstall even if dxstream/pydxs already appear installed.

set -u

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEMO_DIR=$(realpath -s "${SCRIPT_DIR}/..")

# Optional colour helpers (fall back to plain echo when unavailable).
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

# --- Configuration -----------------------------------------------------------
DX_STREAM_REPO="${DX_STREAM_REPO:-git@github.com:DEEPX-AI/dx_stream}"
PREFIX="/usr/local"
SKIP_DEPS=0
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --skip-deps) SKIP_DEPS=1 ;;
        --force)     FORCE=1 ;;
        --prefix=*)  PREFIX="${arg#*=}" ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) err "Unknown option: $arg"; exit 1 ;;
    esac
done

# Pick the interpreter the demo will actually run with.
PYBIN="${PYTHON:-}"
if [ -z "${PYBIN}" ]; then
    if command -v python >/dev/null 2>&1; then PYBIN="python"; else PYBIN="python3"; fi
fi

# --- Helpers -----------------------------------------------------------------
pydxs_importable() {
    "${PYBIN}" -c "import pydxs" >/dev/null 2>&1
}

dxstream_plugin_available() {
    # Prefer gi (matches the demo's runtime check); fall back to gst-inspect.
    "${PYBIN}" - <<'PYEOF' >/dev/null 2>&1 && return 0
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
if not Gst.is_initialized():
    Gst.init(None)
import sys
sys.exit(0 if Gst.ElementFactory.find("dxpreprocess") else 1)
PYEOF
    command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 dxpreprocess >/dev/null 2>&1
}

already_installed() {
    pydxs_importable && dxstream_plugin_available
}

dxrt_available() {
    "${PYBIN}" -c "import dx_engine" >/dev/null 2>&1
}

locate_source() {
    local candidates=(
        "${DX_STREAM_SRC:-}"
        "${DEMO_DIR}/../dx_stream"
        "${DEMO_DIR}/../../dx_stream"
        "${HOME}/workspace/dx_stream"
        "${HOME}/dx_stream"
    )
    local c
    for c in "${candidates[@]}"; do
        [ -z "$c" ] && continue
        if [ -f "${c}/build.sh" ] && [ -d "${c}/gst-dxstream-plugin" ]; then
            realpath "$c"
            return 0
        fi
    done
    return 1
}

clone_source() {
    local dest="${DEMO_DIR}/third_party/dx_stream"
    if [ -f "${dest}/build.sh" ]; then
        realpath "$dest"
        return 0
    fi
    if ! command -v git >/dev/null 2>&1; then
        err "git not found; cannot clone dx_stream."
        hint "Install git, or set DX_STREAM_SRC to an existing dx_stream checkout."
        return 1
    fi
    log "Cloning dx_stream from ${DX_STREAM_REPO} into ${dest}..."
    mkdir -p "${DEMO_DIR}/third_party"
    if git clone --depth 1 "${DX_STREAM_REPO}" "${dest}" >&2; then
        realpath "$dest"
        return 0
    fi
    err "Failed to clone dx_stream from ${DX_STREAM_REPO}."
    hint "Set DX_STREAM_REPO (e.g. https URL) or DX_STREAM_SRC to a local checkout."
    return 1
}

# --- Main --------------------------------------------------------------------
log "=== dx_stream (dxstream backend) installation ==="

if [ "$FORCE" -ne 1 ] && already_installed; then
    ok "dx_stream plugin and pydxs are already installed. Nothing to do."
    hint "Re-run with --force to rebuild from source."
    exit 0
fi

if ! dxrt_available; then
    warn "DX-RT (dx_engine) is not importable. dx_stream links against DX-RT and"
    warn "its build will fail without it."
    hint "Install/build DX-RT first (see the demo Prerequisites), then re-run this script."
    # Continue anyway so the user sees the concrete build error if they insist,
    # but make the cause obvious up front.
fi

SRC="$(locate_source)" || SRC=""
if [ -z "$SRC" ]; then
    log "No local dx_stream checkout found; attempting to clone..."
    SRC="$(clone_source)" || {
        err "Could not obtain dx_stream source."
        exit 1
    }
fi
ok "Using dx_stream source: ${SRC}"

pushd "${SRC}" >/dev/null || { err "Cannot enter ${SRC}"; exit 1; }

if [ "$SKIP_DEPS" -ne 1 ]; then
    log "Installing dx_stream system dependencies (install.sh)..."
    if ! ./install.sh; then
        err "dx_stream install.sh failed."
        hint "Inspect the output above, or re-run with --skip-deps if deps are already present."
        popd >/dev/null
        exit 1
    fi
else
    log "Skipping dx_stream system dependencies (--skip-deps)."
fi

log "Building & installing dx_stream plugin + pydxs (build.sh --prefix=${PREFIX})..."
if ! ./build.sh --prefix="${PREFIX}"; then
    err "dx_stream build.sh failed."
    popd >/dev/null
    exit 1
fi

popd >/dev/null

# Resolve the installed GStreamer plugin directory and persist a sourced env
# file so run_demo.sh works without requiring `source ~/.bashrc` first.
PLUGIN_SO="$(find "${PREFIX}/lib" -name libgstdxstream.so -path '*/gstreamer-1.0/*' 2>/dev/null | head -n1)"
if [ -n "${PLUGIN_SO}" ]; then
    PLUGIN_DIR="$(dirname "${PLUGIN_SO}")"
    LIBDIR="$(dirname "${PLUGIN_DIR}")"
    ENV_FILE="${SCRIPT_DIR}/.dxstream_env.sh"
    cat > "${ENV_FILE}" <<EOF
# Auto-generated by install_dxstream.sh. Sourced by run_demo.sh so the dxstream
# backend can locate the dx_stream GStreamer plugin without editing ~/.bashrc.
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH}"
export GST_PLUGIN_PATH="${PLUGIN_DIR}:\${GST_PLUGIN_PATH}"
export LD_LIBRARY_PATH="${PLUGIN_DIR}:${PREFIX}/share/gstdxstream/lib:\${LD_LIBRARY_PATH}"
export PATH="${PREFIX}/share/gstdxstream/bin:\${PATH}"
EOF
    ok "Wrote environment file: ${ENV_FILE}"
    log "run_demo.sh will source it automatically."
else
    warn "Could not locate libgstdxstream.so under ${PREFIX}/lib."
    warn "build.sh should have added GST_PLUGIN_PATH to ~/.bashrc; run 'source ~/.bashrc'."
fi

# Verify (in the same shell, using the freshly written env if present).
if [ -f "${SCRIPT_DIR}/.dxstream_env.sh" ]; then
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/.dxstream_env.sh"
fi
if already_installed; then
    ok "dx_stream installation verified (plugin + pydxs available)."
    exit 0
fi

warn "Installation finished but verification did not pass in this shell."
hint "Open a new terminal (or 'source ~/.bashrc') and re-run the demo with engine_backend: dxstream."
exit 0
