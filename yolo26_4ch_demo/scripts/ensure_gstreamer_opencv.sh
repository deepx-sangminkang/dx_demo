#!/bin/bash
# Ensure a GStreamer-enabled OpenCV is importable.
#
# The demo offloads video decoding to platform hardware decoders (RK3588
# mppvideodec, Intel VAAPI) and the RGA colour-convert element through
# cv2.VideoCapture(..., cv2.CAP_GSTREAMER). That backend only exists when the
# active OpenCV build was compiled WITH GStreamer support. The PyPI
# `opencv-python` wheel is built WITHOUT GStreamer, so this script enforces a
# GStreamer-enabled OpenCV (or fails the installation with actionable guidance).

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
warn() { print_colored "$1" "WARNING"; }
err()  { print_colored "$1" "ERROR"; }

# Pick the interpreter the demo will actually run with.
PYBIN="${PYTHON:-}"
if [ -z "${PYBIN}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYBIN="python"
    else
        PYBIN="python3"
    fi
fi

# Probe the active interpreter for venv / OpenCV / GStreamer facts.
# Prints `key=value` lines consumed below.
probe_env() {
    "${PYBIN}" - <<'PYEOF'
import sys

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

system_site = True
if in_venv:
    cfg = None
    import os
    candidate = os.path.join(sys.prefix, "pyvenv.cfg")
    system_site = False
    try:
        with open(candidate) as f:
            for line in f:
                k, _, v = line.partition("=")
                if k.strip() == "include-system-site-packages":
                    system_site = v.strip().lower() == "true"
    except OSError:
        system_site = False

has_cv2 = False
gstreamer = False
try:
    import cv2  # noqa: F401
    has_cv2 = True
    for raw in cv2.getBuildInformation().splitlines():
        s = raw.strip()
        if s.startswith("GStreamer:"):
            gstreamer = s.split(":", 1)[1].strip().upper().startswith("YES")
            break
except Exception:
    has_cv2 = False

print("in_venv=%s" % int(in_venv))
print("system_site=%s" % int(system_site))
print("has_cv2=%s" % int(has_cv2))
print("gstreamer=%s" % int(gstreamer))
PYEOF
}

read_probe() {
    local out
    out="$(probe_env 2>/dev/null)"
    IN_VENV=0; SYSTEM_SITE=0; HAS_CV2=0; GSTREAMER=0
    # shellcheck disable=SC2046
    eval "$(echo "${out}" | sed -n 's/^\(in_venv\|system_site\|has_cv2\|gstreamer\)=\([01]\)$/\U\1\E=\2/p')"
}

apt_install_opencv() {
    local sudo=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo="sudo"
        else
            err "Root privileges (or sudo) are required to install python3-opencv via apt."
            return 1
        fi
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        err "apt-get not found; cannot auto-install a GStreamer-enabled OpenCV on this distro."
        return 1
    fi

    log "Installing system OpenCV (GStreamer-enabled) and GStreamer plugins via apt..."
    ${sudo} apt-get update -y || { err "apt-get update failed."; return 1; }
    ${sudo} apt-get install -y \
        python3-opencv \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        || { err "apt-get install failed."; return 1; }
    return 0
}

remove_pip_opencv() {
    # The GStreamer-less PyPI wheel shadows the system OpenCV; remove it so the
    # system build is the one that gets imported.
    log "Removing GStreamer-less pip OpenCV wheels (if any)..."
    "${PYBIN}" -m pip uninstall -y \
        opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
}

print_manual_guidance() {
    err "Could not obtain a GStreamer-enabled OpenCV automatically."
    cat <<'EOF'

To enable hardware decoding (decode: hw/auto), OpenCV MUST be built with
GStreamer support. Options:

  1) Use the distro OpenCV (built with GStreamer):
       sudo apt-get install python3-opencv gstreamer1.0-plugins-{base,good,bad,ugly} gstreamer1.0-libav
       pip uninstall -y opencv-python opencv-python-headless

  2) If you use a virtualenv, recreate it so it can see the system OpenCV:
       python3 -m venv --system-site-packages .venv

  3) Or build OpenCV from source with -DWITH_GSTREAMER=ON.

Verify with:
  python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
  (must print 'GStreamer: YES')
EOF
}

main() {
    log "Verifying GStreamer-enabled OpenCV (required for HW decoding)..."
    read_probe

    if [ "${HAS_CV2}" = "1" ] && [ "${GSTREAMER}" = "1" ]; then
        log "[OK] OpenCV has GStreamer support."
        return 0
    fi

    if [ "${HAS_CV2}" = "1" ]; then
        warn "OpenCV is installed but built WITHOUT GStreamer support."
    else
        warn "OpenCV is not importable by '${PYBIN}'."
    fi

    # An isolated virtualenv cannot see the apt-installed system OpenCV, so the
    # automatic remedy would strand the environment without any cv2.
    if [ "${IN_VENV}" = "1" ] && [ "${SYSTEM_SITE}" != "1" ]; then
        err "Active virtualenv was created WITHOUT --system-site-packages."
        print_manual_guidance
        return 1
    fi

    log "Attempting to install a GStreamer-enabled OpenCV automatically..."
    if apt_install_opencv; then
        remove_pip_opencv
        read_probe
        if [ "${HAS_CV2}" = "1" ] && [ "${GSTREAMER}" = "1" ]; then
            log "[OK] GStreamer-enabled OpenCV is now active."
            return 0
        fi
    fi

    print_manual_guidance
    return 1
}

main "$@"
