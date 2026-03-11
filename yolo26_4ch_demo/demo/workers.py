"""Multi-channel YOLO26 pipeline worker/thread skeleton.

Skeleton code implementing the C-style architecture (per-channel capture + global pre/infer/draw workers).
- Keeps the actual logic concise; comments written to clearly convey each component's role.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections import defaultdict

import cv2
import numpy as np

from .engine import YOLO26Engine


# ===== Per-stage drop counts (simple/intuitive version) =====

# queue_drop_counts[stage][channel_id] = count
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


# ===== Data structures for shared queues =====


@dataclass
class CaptureItem:
    """Data passed from capture thread to preprocess_worker.

    channel_id: identifies which channel the frame came from
    frame_bgr: original BGR frame (kept for visualisation)
    meta: supplementary information such as timestamps
    """

    channel_id: int
    frame_bgr: np.ndarray
    meta: Dict[str, Any]


@dataclass
class InferItem:
    """Data passed from preprocess_worker to wait_worker."""

    channel_id: int
    frame_bgr: np.ndarray
    input_tensor: np.ndarray
    req_id: int
    meta: Dict[str, Any]


@dataclass
class OutputItem:
    """Data passed from wait_worker to draw_worker."""

    channel_id: int
    frame_bgr: np.ndarray
    output_tensors: Any
    meta: Dict[str, Any]


# ===== Per-channel capture threads =====


class CaptureThread(threading.Thread):
    """Capture thread created once per channel.

    - Reads frames from USB Cam / video file / RTSP and puts them into the shared input_queue.
    - DX inference and GUI updates are handled by other workers/threads.
    """

    def __init__(
        self,
        channel_id: int,
        source: Any,
        input_queue: "queue.Queue[CaptureItem]",
        max_fps: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(daemon=True, name=name or f"CaptureThread-{channel_id}")
        self.channel_id = channel_id
        self.source = source
        self.input_queue = input_queue
        self.max_fps = max_fps
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request thread shutdown from outside."""

        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - runtime only
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Channel {self.channel_id}: cannot open input source - {self.source}")
            return

        print(f"[INFO] Channel {self.channel_id}: capture started - {self.source}")

        min_interval = 1.0 / self.max_fps if self.max_fps and self.max_fps > 0 else 0.0

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()
                if not ok:
                    # For video files, seek back to the beginning when the end is reached for infinite looping.
                    # For RTSP/camera sources there is no EOF concept, or it may indicate an error,
                    # but since file path (str) input is the typical demo scenario, handle it as a file for now.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        print(
                            f"[INFO] Channel {self.channel_id}: no more frames available (EOF or error)"
                        )
                        break

                meta: Dict[str, Any] = {
                    "t_read": time.perf_counter() - t0,
                    "ts": time.time(),
                }

                item = CaptureItem(
                    channel_id=self.channel_id,
                    frame_bgr=frame_bgr,
                    meta=meta,
                )

                # When the queue is full, drop the oldest frame and insert the latest one
                # to maintain "real-time" behaviour. This way, even when input FPS > processing FPS,
                # stuttering is reduced and the display always shows the most recent frame available.
                try:
                    self.input_queue.put(item, timeout=0.001)
                except queue.Full:
                    try:
                        # Drop one (remove the oldest frame)
                        dropped_item = self.input_queue.get_nowait()
                        # Increment dropped frame count for the capture stage
                        with queue_drop_lock:
                            queue_drop_counts["input"][self.channel_id] += 1
                    except queue.Empty:
                        dropped_item = None

                    try:
                        # Insert the latest frame.
                        self.input_queue.put_nowait(item)
                    except queue.Full:
                        # In extreme cases, silently skip (no log)
                        pass

                # Record read-stage throughput (only when successfully enqueued)
                record_throughput("read", time.time())

                # Apply a simple sleep if FPS cap is configured
                if min_interval > 0.0:
                    elapsed = time.perf_counter() - t0
                    remain = min_interval - elapsed
                    if remain > 0:
                        time.sleep(remain)
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

        if item.frame_bgr is None:
            continue

        t0 = time.perf_counter()
        input_tensor, meta_pre = engine.preprocess(item.frame_bgr)
        item.meta.update(meta_pre)
        item.meta["t_preprocess"] = time.perf_counter() - t0

        # Consider preprocessing complete when run_async is also called.
        req_id = engine.run_async(input_tensor)
        infer_item = InferItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            input_tensor=input_tensor,
            req_id=req_id,
            meta=item.meta,
        )

        # When infer_queue is full, drop the oldest item and insert the latest.
        try:
            infer_queue.put(infer_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = infer_queue.get_nowait()
                # Increment dropped frame count for preprocess stage (infer_queue)
                with queue_drop_lock:
                    queue_drop_counts["infer"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                infer_queue.put_nowait(infer_item)
            except queue.Full:
                # In extreme cases, silently skip (no log)
                pass

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

        t0 = time.perf_counter()
        output_tensors = engine.wait(item.req_id)
        item.meta["t_inference"] = time.perf_counter() - t0

        out_item = OutputItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            output_tensors=output_tensors,
            meta=item.meta,
        )

        # When draw_queue is also full, drop the oldest item and insert the latest.
        try:
            draw_queue.put(out_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = draw_queue.get_nowait()
                # Increment dropped frame count for draw stage (draw_queue)
                with queue_drop_lock:
                    queue_drop_counts["draw"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                draw_queue.put_nowait(out_item)
            except queue.Full:
                # In extreme cases, silently skip (no log)
                pass

        # Record inference-stage throughput (after wait completes and item is put into draw_queue)
        record_throughput("inf", time.time())


def draw_worker(
    engine: YOLO26Engine,
    draw_queue: "queue.Queue[OutputItem]",
    get_selected_classes,
    on_frame_ready,
    stop_event: threading.Event,
) -> None:
    """Global draw + GUI delivery worker.

    - Receives frame + meta(detections) from draw_queue,
      calls draw_detections, and forwards to on_frame_ready.
    """

    last_log_ts = time.time()

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = draw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detections = np.squeeze(item.output_tensors)

        # Read the currently selected class set and filter
        selected_classes = get_selected_classes()

        if selected_classes is None:
            filtered_detections = detections
        elif len(selected_classes) == 0:
            # No class selected → do not draw any boxes
            filtered_detections = np.empty((0, 6), dtype=detections.dtype)
        else:
            cls_mask = np.isin(detections[:, 5].astype(int), list(selected_classes))
            filtered_detections = detections[cls_mask]

        t_draw0 = time.perf_counter()
        engine.draw_detections(item.frame_bgr, filtered_detections, item.meta)
        t_draw1 = time.perf_counter()

        item.meta["t_draw"] = t_draw1 - t_draw0

        on_frame_ready(item.channel_id, item.frame_bgr, item.meta)

        # Record draw-stage throughput (at the point frame visualisation and GUI delivery completes)
        now = time.time()
        record_throughput("draw", now)
