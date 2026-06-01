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
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.realpath(__file__))
DEMO_DIR = os.path.dirname(HERE)
DEFAULT_VIDEO = os.path.join(DEMO_DIR, "assets", "videos", "carrierbag.mp4")

APPSINK = "appsink drop=true max-buffers=1 sync=false"
DRAIN = "src. ! queue ! fakesink async=false sync=false"

# A genuinely-working HW pipeline prerolls and delivers a frame in well under a
# second; a broken one (e.g. dxconvert stuck in PAUSED) blocks forever. We run
# each variant in its own subprocess and hard-kill it after this many seconds so
# one hanging/segfaulting variant cannot stop us from testing the rest.
PER_VARIANT_TIMEOUT_S = 8


def variants(video: str):
    """(name, pipeline) candidates.

    Ordered so the variants most likely to *work* and *least* likely to hang
    (plain ``videoconvert``) run first on a clean VPU; the ``dxconvert`` variants
    (one of which deadlocks in PAUSED) run last. A killed/hung mpp+RGA session
    can wedge the VPU so that every later ``mppvideodec`` pipeline then fails to
    construct, which is exactly why the dangerous ones must come last.
    """

    return [
        (
            "BGR + videoconvert + audio-drain (best candidate for the demo)",
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
            "RGB + dxconvert + audio-drain (current demo HW path; known to HANG)",
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
            "RGB + dxconvert, NO audio-drain (original)",
            f"filesrc location={video} ! parsebin ! mppvideodec ! dxconvert "
            f"! video/x-raw,format=RGB ! {APPSINK}",
        ),
    ]


def _probe_single(index: int, video: str, want_frames: int = 5) -> int:
    """Open one variant, try to read a few frames, print a RESULT line, exit.

    Runs as a short-lived child process so the parent can hard-kill it if the
    pipeline deadlocks (read() has no honoured timeout on older OpenCV builds).
    """

    name, pipeline = variants(video)[index]
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    for prop_name, value in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
        ("CAP_PROP_READ_TIMEOUT_MSEC", 3000),
    ):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            try:
                cap.set(prop, value)
            except Exception:
                pass
    if not cap.isOpened():
        print("RESULT: isOpened() == False (pipeline failed to construct)", flush=True)
        return 0
    got = 0
    frame = None
    t0 = time.time()
    for _ in range(want_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        got += 1
    dt = time.time() - t0
    if got:
        h, w = frame.shape[:2]
        print(
            f"RESULT: OK - read {got}/{want_frames} frames ({w}x{h}) in "
            f"{dt:.2f}s  <<< THIS PIPELINE WORKS",
            flush=True,
        )
    else:
        print(f"RESULT: opened but 0 frames in {dt:.2f}s (no buffers)", flush=True)
    # Intentionally do NOT cap.release() a possibly-stuck pipeline: releasing a
    # PAUSED dxconvert/mpp pipeline can segfault. We're a throwaway child, so let
    # the OS reclaim everything on exit instead.
    os._exit(0)


def _run_variant(i: int, video: str, working: list) -> bool:
    """Run variant ``i`` in an isolated child. Returns True if it hung (which
    can wedge the VPU for subsequent variants)."""

    name, pipeline = variants(video)[i]
    print("\n" + "=" * 70)
    print(f" [{i}] {name}")
    print(f"   {pipeline}")
    print("-" * 70)
    try:
        proc = subprocess.run(
            [sys.executable, os.path.realpath(__file__), "--single", str(i), video],
            timeout=PER_VARIANT_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "").strip()
        if partial:
            print(f"   {partial}")
        print(
            f"   RESULT: HANG - no frames within {PER_VARIANT_TIMEOUT_S}s "
            f"(pipeline stuck in PAUSED) -> SKIP"
        )
        print(
            "   NOTE: a hung mpp/RGA pipeline can wedge the VPU so later "
            "variants fail to construct."
        )
        return True

    out = (proc.stdout or "").strip()
    if out:
        for line in out.splitlines():
            print(f"   {line}")
    if "THIS PIPELINE WORKS" in out:
        working.append((i, name))
    if proc.returncode and proc.returncode < 0:
        print(
            f"   RESULT: child crashed (signal {-proc.returncode}) "
            f"-> pipeline unstable in OpenCV"
        )
    err = (proc.stderr or "").strip()
    if err:
        tail = [l for l in err.splitlines() if l.strip()][-2:]
        for line in tail:
            print(f"   [stderr] {line}")
    return False


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--single":
        index = int(argv[1])
        video = argv[2]
        return _probe_single(index, video)

    # ``--only N`` tests a single variant in isolation on a clean VPU (use this
    # after a HANG wedged the VPU, ideally right after a reboot).
    only_index = None
    if argv and argv[0] == "--only":
        only_index = int(argv[1])
        argv = argv[2:]

    video = argv[0] if argv else DEFAULT_VIDEO
    if not os.path.isfile(video):
        print(f"[ERROR] video not found: {video}", file=sys.stderr)
        return 1

    info = cv2.getBuildInformation()
    gst = next((l.strip() for l in info.splitlines()
                if l.strip().startswith("GStreamer:")), "GStreamer: ?")
    print(f"OpenCV {cv2.__version__} | {gst}")
    print(f"Video: {video}")
    print(f"(each variant runs in its own process, hard-killed after "
          f"{PER_VARIANT_TIMEOUT_S}s if it hangs)")

    working: list = []
    n = len(variants(video))
    if only_index is not None:
        if not 0 <= only_index < n:
            print(f"[ERROR] --only index out of range (0..{n - 1})", file=sys.stderr)
            return 1
        _run_variant(only_index, video, working)
    else:
        for i in range(n):
            _run_variant(i, video, working)

    print("\n" + "=" * 70)
    if working:
        print(" WORKING variant(s):")
        for i, name in working:
            print(f"   [{i}] {name}")
        print(" -> Use this colour/convert path for the demo's HW decode.")
    else:
        print(" No variant produced frames in OpenCV on this run.")
        print(" If the FIRST dxconvert variant HUNG, it may have wedged the VPU")
        print(" and poisoned the others -- reboot the board and retest a single")
        print(" videoconvert variant in isolation, e.g.:")
        print("     python scripts/diagnose_opencv_gst.py --only 0")
        print(" If even that fails, HW decode is not usable through OpenCV's")
        print(" appsink here -> keep SW decode (set decode: \"sw\").")
    print(" Share this whole output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
