"""Multi-channel YOLO26 pipeline worker/thread skeleton.

Skeleton code implementing the C-style architecture (per-channel capture + global pre/infer/draw workers).
- Keeps the actual logic concise; comments written to clearly convey each component's role.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np

from .engine import YOLO26Engine
from . import gst_pipeline as gst


# ===== Per-stage drop counts (simple/intuitive version) =====

queue_drop_counts: Dict[str, Dict[int, int]] = {
    "input": defaultdict(int),      # input_queue (capture stage)
    "infer": defaultdict(int),      # infer_queue (preprocess stage)
    "draw": defaultdict(int),       # draw_queue (draw stage)
}
queue_drop_lock = threading.Lock()


# ===== Per-stage throughput statistics =====

throughput_stats: Dict[str, Dict[str, Any]] = {
    "read": {"first_ts": None, "last_ts": None, "count": 0},
    "pre": {"first_ts": None, "last_ts": None, "count": 0},
    "inf": {"first_ts": None, "last_ts": None, "count": 0},
    "draw": {"first_ts": None, "last_ts": None, "count": 0},
}
throughput_lock = threading.Lock()


def record_throughput(stage: str, ts: float) -> None:
    """Update per-stage throughput statistics.

    - Records the first processing timestamp (first_ts), last timestamp (last_ts), and processed frame count (count).
    """

    with throughput_lock:
        s = throughput_stats.get(stage)
        if s is None:
            return
        if s["first_ts"] is None:
            s["first_ts"] = ts
        s["last_ts"] = ts
        s["count"] += 1


def get_fps(stage: str) -> float:
    """Calculate FPS for the given stage from stored statistics."""

    with throughput_lock:
        s = throughput_stats.get(stage)
        if not s:
            return 0.0
        first_ts = s["first_ts"]
        last_ts = s["last_ts"]
        count = s["count"]

    if first_ts is None or last_ts is None or last_ts <= first_ts or count <= 0:
        return 0.0

    return count / (last_ts - first_ts)


def _increment_drop_count(stage: str, channel_id: Optional[int]) -> None:
    """Increment the dropped-item counter for a queue stage."""

    if channel_id is None:
        return

    with queue_drop_lock:
        queue_drop_counts[stage][channel_id] += 1


def _queue_item_channel_id(item: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Extract a channel id from a queued item when available."""

    if item is None:
        return fallback
    return getattr(item, "channel_id", fallback)


def _enqueue_latest(
    target_queue: "queue.Queue[Any]",
    item: Any,
    stage: str,
    fallback_channel_id: Optional[int] = None,
) -> None:
    """Prefer the newest item by dropping one oldest item when the queue is full."""

    try:
        target_queue.put(item, timeout=0.001)
        return
    except queue.Full:
        pass

    dropped_item = None
    try:
        dropped_item = target_queue.get_nowait()
    except queue.Empty:
        dropped_item = None

    _increment_drop_count(stage, _queue_item_channel_id(dropped_item, fallback_channel_id))

    try:
        target_queue.put_nowait(item)
    except queue.Full:
        pass


# ===== Data structures for shared queues =====


@dataclass
class CaptureItem:
    """Data passed from capture thread to preprocess_worker.

    channel_id: identifies which channel the frame came from
    frame_bgr: original captured frame (BGR by default; RGB when the RGA
        dxconvert HW path is active - see ``meta["color_format"]``)
    meta: supplementary information such as timestamps and ``color_format``
    """

    channel_id: int
    frame_bgr: np.ndarray
    meta: Dict[str, Any]


@dataclass
class InferItem:
    """Data passed from preprocess_worker to wait_worker."""

    channel_id: int
    input_tensor: np.ndarray
    req_id: int
    meta: Dict[str, Any]


@dataclass
class OutputItem:
    """Data passed from wait_worker to detect_worker."""

    channel_id: int
    output_tensors: Any
    meta: Dict[str, Any]


# ===== Per-channel capture threads =====


_hw_decode_env_lock = threading.Lock()
_hw_decode_env: Optional[Dict[str, Any]] = None


def _get_hw_decode_env() -> Dict[str, Any]:
    """Detect platform / OpenCV GStreamer support once and cache the result."""

    global _hw_decode_env
    with _hw_decode_env_lock:
        if _hw_decode_env is None:
            platform = gst.detect_platform()
            _hw_decode_env = {
                "platform": platform,
                "opencv_gstreamer": gst.opencv_has_gstreamer(),
                "rga_convert": gst.rga_convert_available(platform),
            }
        return _hw_decode_env


class CaptureThread(threading.Thread):
    """Capture thread created once per channel.

    - Reads frames from USB Cam / video file / RTSP and puts them into the shared input_queue.
    - DX inference and GUI updates are handled by other workers/threads.
    - When ``decode_mode`` allows and the platform/OpenCV support HW decoding,
      frames are decoded through a GStreamer HW pipeline; otherwise it falls
      back to the default software ``cv2.VideoCapture`` path.
    """

    def __init__(
        self,
        channel_id: int,
        source: Any,
        input_queue: "queue.Queue[CaptureItem]",
        max_fps: Optional[float] = None,
        name: Optional[str] = None,
        source_type: str = "video",
        decode_mode: str = "auto",
        display_callback: Optional[Callable[[int, np.ndarray, str], None]] = None,
        scale_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(daemon=True, name=name or f"CaptureThread-{channel_id}")
        self.channel_id = channel_id
        self.source = source
        self.input_queue = input_queue
        self.max_fps = max_fps
        self.source_type = source_type or "video"
        self.decode_mode = decode_mode or "auto"
        # Optional callback invoked for every captured frame so the display can
        # update at capture FPS, decoupled from the (slower) inference pipeline.
        self.display_callback = display_callback
        # When set, request RGA HW resize (dxscale) to this (w, h) so the CPU
        # cv2.resize is skipped; only effective on the RK3588 RGB decode path.
        self.scale_size = scale_size
        self.used_hw = False
        # Colour order of frames this thread produces ("bgr" or "rgb"); the RGA
        # dxconvert HW path on RK3588 yields RGB so downstream cvtColor is skipped.
        self.color_format = gst.COLOR_BGR
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request thread shutdown from outside."""

        self._stop_event.set()

    def _open_capture(self) -> "cv2.VideoCapture":
        """Open the input source, preferring HW decoding with SW fallback."""

        env = _get_hw_decode_env()
        cap, used_hw, reason, color_format = gst.open_capture(
            source=self.source,
            source_type=self.source_type,
            decode_mode=self.decode_mode,
            platform=env["platform"],
            opencv_gstreamer=env["opencv_gstreamer"],
            rga_convert=env["rga_convert"],
            scale_size=self.scale_size,
        )
        self.used_hw = used_hw
        self.color_format = color_format
        decode_kind = "HW (GStreamer)" if used_hw else "SW"
        print(
            f"[INFO] Channel {self.channel_id}: decode={decode_kind} "
            f"color={color_format} ({self.source_type}) - {reason}"
        )
        return cap

    def _read_frame(self, cap: "cv2.VideoCapture") -> Optional[np.ndarray]:
        """Read one frame, rewinding/looping file sources when EOF is reached."""

        ok, frame_bgr = cap.read()
        if ok:
            return frame_bgr

        # Only file (video) sources loop; live sources (rtsp/camera) stop on EOF.
        if self.source_type != "video":
            print(
                f"[INFO] Channel {self.channel_id}: stream ended ({self.source_type})"
            )
            return None

        # Software backend supports in-place seek to the first frame.
        if not self.used_hw:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
            if ok:
                return frame_bgr

        print(f"[INFO] Channel {self.channel_id}: no more frames available (EOF or error)")
        return None

    def _enqueue_capture_item(self, item: CaptureItem) -> None:
        """Push the newest capture item into the shared input queue."""

        _enqueue_latest(
            target_queue=self.input_queue,
            item=item,
            stage="input",
            fallback_channel_id=self.channel_id,
        )

    def _sleep_for_fps_limit(self, start_ts: float, min_interval: float) -> None:
        """Sleep just enough to respect an optional FPS cap."""

        if min_interval <= 0.0:
            return

        elapsed = time.perf_counter() - start_ts
        remain = min_interval - elapsed
        if remain > 0:
            time.sleep(remain)

    def run(self) -> None:  # pragma: no cover - runtime only
        cap = self._open_capture()
        if not cap.isOpened():
            print(f"[ERROR] Channel {self.channel_id}: cannot open input source - {self.source}")
            return

        print(f"[INFO] Channel {self.channel_id}: capture started - {self.source}")

        min_interval = 1.0 / self.max_fps if self.max_fps and self.max_fps > 0 else 0.0

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                frame_bgr = self._read_frame(cap)
                if frame_bgr is None:
                    # HW (GStreamer) backend cannot seek; reopen to loop video files.
                    if self.used_hw and self.source_type == "video":
                        cap.release()
                        cap = self._open_capture()
                        if not cap.isOpened():
                            break
                        continue
                    break

                meta: Dict[str, Any] = {
                    "t_read": time.perf_counter() - t0,
                    "ts": time.time(),
                    "color_format": self.color_format,
                }

                item = CaptureItem(
                    channel_id=self.channel_id,
                    frame_bgr=frame_bgr,
                    meta=meta,
                )

                self._enqueue_capture_item(item)

                # Forward the captured frame straight to the display path so the
                # screen refreshes at capture FPS, independent of inference speed.
                if self.display_callback is not None:
                    self.display_callback(self.channel_id, frame_bgr, self.color_format)

                # Record read-stage throughput (only when successfully enqueued)
                record_throughput("read", time.time())

                self._sleep_for_fps_limit(t0, min_interval)
        finally:
            cap.release()
            print(f"[INFO] Channel {self.channel_id}: capture stopped")


# ===== Global worker thread functions =====


def preprocess_worker(
    engine: YOLO26Engine,
    input_queue: "queue.Queue[CaptureItem]",
    infer_queue: "queue.Queue[InferItem]",
    stop_event: threading.Event,
) -> None:
    """Global preprocess + run_async worker.

    - Receives frames from multiple channels via a single queue and processes them
    - After preprocess, calls run_async and forwards the req_id to infer_queue
    """

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        if item.frame_bgr is None:
            continue

        t0 = time.perf_counter()
        color_format = item.meta.get("color_format", "bgr")
        input_tensor, meta_pre = engine.preprocess(item.frame_bgr, color_format)
        item.meta.update(meta_pre)
        item.meta["t_preprocess"] = time.perf_counter() - t0

        # Consider preprocessing complete when run_async is also called.
        req_id = engine.run_async(input_tensor)
        infer_item = InferItem(
            channel_id=item.channel_id,
            input_tensor=input_tensor,
            req_id=req_id,
            meta=item.meta,
        )

        _enqueue_latest(infer_queue, infer_item, stage="infer")

        # Record preprocess-stage throughput (at the point preprocess + run_async completes)
        record_throughput("pre", time.time())


def wait_worker(
    engine: YOLO26Engine,
    infer_queue: "queue.Queue[InferItem]",
    draw_queue: "queue.Queue[OutputItem]",
    stop_event: threading.Event,
) -> None:
    """Global wait/inference worker.

    - Waits for the req_id sent via run_async, then forwards output_tensors to draw_queue
    """

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = infer_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        t0 = time.perf_counter()
        output_tensors = engine.wait(item.req_id)
        item.meta["t_inference"] = time.perf_counter() - t0

        out_item = OutputItem(
            channel_id=item.channel_id,
            output_tensors=output_tensors,
            meta=item.meta,
        )

        _enqueue_latest(draw_queue, out_item, stage="draw")

        # Record inference-stage throughput (after wait completes and item is put into draw_queue)
        record_throughput("inf", time.time())


def detect_worker(
    engine: YOLO26Engine,
    draw_queue: "queue.Queue[OutputItem]",
    get_selected_classes,
    on_detections_ready,
    stop_event: threading.Event,
) -> None:
    """Global detection post-processing + GUI delivery worker.

    - Receives inference outputs + meta from draw_queue,
      computes filtered detections in original-image coordinates, and forwards
      them to ``on_detections_ready`` (no frame is touched here; the GUI overlays
      boxes onto the separately-displayed capture frame).
    """

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = draw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        selected_classes = get_selected_classes()

        t_draw0 = time.perf_counter()
        detections = engine.finalize_detections(
            item.output_tensors, item.meta, selected_classes
        )
        t_draw1 = time.perf_counter()

        item.meta["t_draw"] = t_draw1 - t_draw0

        on_detections_ready(item.channel_id, detections, item.meta)

        # Record draw-stage throughput (after detections are computed and delivered)
        now = time.time()
        record_throughput("draw", now)
