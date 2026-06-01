#!/usr/bin/env python3
"""Probe which GStreamer pipeline variants actually yield frames through
OpenCV's ``cv2.VideoCapture(..., cv2.CAP_GSTREAMER)``.

``gst-launch-1.0`` can succeed while OpenCV still gets zero frames, because the
OpenCV appsink negotiates its own caps (it strongly prefers BGR) and links the
pipeline differently. This script opens each candidate pipeline exactly the way
the demo does and reports how many frames it can read, so we can tell whether
the blocker is the RGB caps, the audio multiqueue, or something else.

Run on the target device with the demo's interpreter:
    python scripts/diagnose_opencv_gst.py [VIDEO_FILE]
"""

from __future__ import annotations

import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.realpath(__file__))
DEMO_DIR = os.path.dirname(HERE)
DEFAULT_VIDEO = os.path.join(DEMO_DIR, "assets", "videos", "carrierbag.mp4")

APPSINK = "appsink drop=true max-buffers=1 sync=false"
DRAIN = "src. ! queue ! fakesink async=false sync=false"


def variants(video: str):
    """(name, pipeline) candidates, from most to least hardware-offloaded."""

    return [
        (
            "RGB + dxconvert + audio-drain (current demo HW path)",
            f"filesrc location={video} ! parsebin name=src "
            f"src. ! mppvideodec ! dxconvert ! video/x-raw,format=RGB ! {APPSINK} "
            f"{DRAIN}",
        ),
        (
            "BGR + dxconvert + audio-drain",
            f"filesrc location={video} ! parsebin name=src "
            f"src. ! mppvideodec ! dxconvert ! video/x-raw,format=BGR ! {APPSINK} "
            f"{DRAIN}",
        ),
        (
            "BGR + videoconvert + audio-drain (RGA-backed convert, OpenCV-friendly)",
            f"filesrc location={video} ! parsebin name=src "
            f"src. ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! {APPSINK} "
            f"{DRAIN}",
        ),
        (
            "BGR + videoconvert, NO audio-drain",
            f"filesrc location={video} ! parsebin ! mppvideodec ! videoconvert "
            f"! video/x-raw,format=BGR ! {APPSINK}",
        ),
        (
            "BGR + videoconvert via qtdemux+h264parse (explicit demux)",
            f"filesrc location={video} ! qtdemux ! h264parse ! mppvideodec "
            f"! videoconvert ! video/x-raw,format=BGR ! {APPSINK}",
        ),
        (
            "RGB + dxconvert, NO audio-drain (original)",
            f"filesrc location={video} ! parsebin ! mppvideodec ! dxconvert "
            f"! video/x-raw,format=RGB ! {APPSINK}",
        ),
    ]


def try_pipeline(name: str, pipeline: str, want_frames: int = 5) -> None:
    print("\n" + "=" * 70)
    print(f" {name}")
    print(f"   {pipeline}")
    print("-" * 70)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("   RESULT: isOpened() == False (pipeline failed to construct)")
        return
    got = 0
    t0 = time.time()
    for _ in range(want_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        got += 1
    dt = time.time() - t0
    cap.release()
    if got:
        h, w = frame.shape[:2]
        print(f"   RESULT: OK - read {got}/{want_frames} frames "
              f"({w}x{h}) in {dt:.2f}s  <<< THIS PIPELINE WORKS")
    else:
        print(f"   RESULT: opened but 0 frames in {dt:.2f}s (no buffers)")


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not os.path.isfile(video):
        print(f"[ERROR] video not found: {video}", file=sys.stderr)
        return 1

    info = cv2.getBuildInformation()
    gst = next((l.strip() for l in info.splitlines()
                if l.strip().startswith("GStreamer:")), "GStreamer: ?")
    print(f"OpenCV {cv2.__version__} | {gst}")
    print(f"Video: {video}")

    for name, pipeline in variants(video):
        try:
            try_pipeline(name, pipeline)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"   RESULT: exception {exc!r}")

    print("\n" + "=" * 70)
    print(" Pick the first pipeline marked 'THIS PIPELINE WORKS'. If a BGR")
    print(" videoconvert variant works but the RGB dxconvert ones don't, the")
    print(" demo should use the BGR HW path for OpenCV. Share this output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
