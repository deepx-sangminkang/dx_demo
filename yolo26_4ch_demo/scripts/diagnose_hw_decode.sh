#!/bin/bash
# Pinpoint why the RK3588 hardware-decode GStreamer pipeline produces no frames.
#
# The demo builds (on RK3588 with the dx_stream plugin present):
#   filesrc ! parsebin ! mppvideodec ! dxconvert ! video/x-raw,format=RGB ! appsink
#
# When that pipeline negotiates no caps and emits no buffer, OpenCV prints
# "cannot query video width/height" and the channel reports
# "no more frames available (EOF or error)".
#
# This script runs the pipeline in layers with `gst-launch-1.0 -v` so you can
# see exactly which element fails to link / produce output:
#   1. file + demux/parse (no decode)        -> is the file/parse OK?
#   2. + mppvideodec (HW decode)             -> does the VPU decode?
#   3. + videoconvert  -> BGR appsink        -> CPU colour path (demo SW-ish)
#   4. + dxconvert     -> RGB appsink        -> RGA colour path (demo HW path)
#
# Usage:
#   scripts/diagnose_hw_decode.sh [VIDEO_FILE]
# Defaults to assets/videos/carrierbag.mp4 (H.264, the simplest sample).

set -u

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEMO_DIR=$(realpath -s "${SCRIPT_DIR}/..")
VIDEO="${1:-${DEMO_DIR}/assets/videos/carrierbag.mp4}"

if [ ! -f "${VIDEO}" ]; then
    echo "[ERROR] video not found: ${VIDEO}" >&2
    exit 1
fi
if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
    echo "[ERROR] gst-launch-1.0 not found (install gstreamer1.0-tools)" >&2
    exit 1
fi

# Make the dx_stream plugin discoverable (same as run_demo.sh).
if [ -f "${SCRIPT_DIR}/.dxstream_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.dxstream_env.sh"
fi

echo "==================================================================="
echo " HW decode diagnosis for: ${VIDEO}"
echo "==================================================================="

echo
echo "[elements] presence check:"
for el in parsebin qtdemux h264parse mppvideodec videoconvert dxconvert dxscale; do
    if gst-inspect-1.0 "${el}" >/dev/null 2>&1; then
        echo "  OK   ${el}"
    else
        echo "  MISS ${el}"
    fi
done

run_stage() {
    local title="$1"; shift
    echo
    echo "-------------------------------------------------------------------"
    echo " ${title}"
    echo "   ${*}"
    echo "-------------------------------------------------------------------"
    # num-buffers keeps each stage short; -v shows negotiated caps; non-zero
    # exit or an ERROR line means that stage is the culprit.
    timeout 20 gst-launch-1.0 -v "$@" 2>&1 | grep -Ei "ERROR|WARN|caps =|Setting pipeline|Got EOS|Pipeline is PREROLL|/GstPipeline" | head -40
    echo "   (exit: ${PIPESTATUS[0]})"
}

run_stage "1) file + demux/parse only (no decode)" \
    filesrc location="${VIDEO}" ! parsebin ! fakesink num-buffers=30 -e

run_stage "2) + mppvideodec (HW decode) -> fakesink" \
    filesrc location="${VIDEO}" ! parsebin ! mppvideodec ! fakesink num-buffers=30 -e

run_stage "3) + videoconvert -> BGR appsink (CPU colour, demo SW-ish path)" \
    filesrc location="${VIDEO}" ! parsebin ! mppvideodec ! videoconvert \
    ! video/x-raw,format=BGR ! identity eos-after=30 \
    ! appsink drop=true max-buffers=1 sync=false

run_stage "4) + dxconvert -> RGB appsink (RGA colour, demo HW path)" \
    filesrc location="${VIDEO}" ! parsebin ! mppvideodec ! dxconvert \
    ! video/x-raw,format=RGB ! identity eos-after=30 \
    ! appsink drop=true max-buffers=1 sync=false

run_stage "5) demo HW path + audio drained to fakesink (multiqueue fix)" \
    filesrc location="${VIDEO}" ! parsebin name=pb \
    pb. ! mppvideodec ! dxconvert ! video/x-raw,format=RGB ! identity eos-after=30 \
    ! appsink drop=true max-buffers=1 sync=false \
    pb. ! queue ! fakesink sync=false

echo
echo "==================================================================="
echo " Interpretation"
echo "  - Stage 1 fails -> file/path or demux problem."
echo "  - Stage 2 fails -> mppvideodec / VPU (rockchip-mpp) not working."
echo "  - Stage 3/4 hang (PREROLLING, no NV12) but 5 works -> parsebin's"
echo "      unused audio stream deadlocks the shared multiqueue; the fix is"
echo "      to drain the extra pad to a fakesink (named parsebin)."
echo "  - Stage 3 OK but 4 fails -> dxconvert (RGA) is the culprit:"
echo "      set 'rga' off so the demo uses videoconvert (BGR) instead."
echo "  - All stages OK -> the issue is OpenCV's appsink negotiation;"
echo "      try decode: \"sw\" in demo/config/yolo26_multich.yaml."
echo "==================================================================="
