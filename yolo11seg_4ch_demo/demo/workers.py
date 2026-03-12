"""Multi-channel YOLOv11 pipeline worker/thread skeleton.

Skeleton code for a C-style architecture
(per-channel capture + global pre/infer/post workers).
- The runtime logic stays concise, with comments focused on making each role clear.
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

from .engine import YOLOv11Engine


# ===== Per-stage drop counters (simple and direct version) =====

# queue_drop_counts[stage][channel_id] = count
queue_drop_counts: Dict[str, Dict[int, int]] = {
    "input": defaultdict(int),      # input_queue (capture stage)
    "infer": defaultdict(int),      # infer_queue (preprocess stage)
    "post": defaultdict(int),       # post_queue (postprocess stage)
    "draw": defaultdict(int),       # draw_queue (draw stage)
}
queue_drop_lock = threading.Lock()


# ===== Per-stage throughput stats =====

throughput_stats: Dict[str, Dict[str, Any]] = {
    "read": {"first_ts": None, "last_ts": None, "count": 0},
    "pre": {"first_ts": None, "last_ts": None, "count": 0},
    "inf": {"first_ts": None, "last_ts": None, "count": 0},
    "post": {"first_ts": None, "last_ts": None, "count": 0},
    "draw": {"first_ts": None, "last_ts": None, "count": 0},
}
throughput_lock = threading.Lock()


def record_throughput(stage: str, ts: float) -> None:
    """Update throughput statistics for a pipeline stage.

    - Record the first timestamp, last timestamp, and processed frame count.
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
    """Compute FPS for the given stage from the recorded statistics."""

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


# ===== Data structures pushed into shared queues =====


@dataclass
class CaptureItem:
    """Data passed from a capture thread to preprocess_worker.

    channel_id: identifies which channel produced the frame
    frame_bgr: original BGR frame retained for visualisation
    meta: auxiliary information such as timestamps
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
    """Data passed from wait_worker to postprocess_worker."""

    channel_id: int
    frame_bgr: np.ndarray
    output_tensors: Any
    meta: Dict[str, Any]


# ===== Per-channel capture thread =====


class CaptureThread(threading.Thread):
    """Capture thread created once per channel.

    - Reads frames from a USB camera, video file, or RTSP and pushes them into input_queue.
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
        """Request the thread to stop externally."""

        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - runtime only
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Channel {self.channel_id}: could not open input source - {self.source}")
            return

        print(f"[INFO] Channel {self.channel_id}: capture started - {self.source}")

        min_interval = 1.0 / self.max_fps if self.max_fps and self.max_fps > 0 else 0.0

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()
                if not ok:
                    # For video files, rewind to the beginning to loop forever at EOF.
                    # RTSP/camera inputs do not really have EOF semantics and this may signal an error,
                    # but file-path input is the typical demo scenario, so prioritise file behaviour here.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        print(
                            f"[INFO] Channel {self.channel_id}: no more frames can be read (EOF or error)"
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

                # If the queue is full, drop the oldest frame and push the newest one
                # to preserve real-time behaviour. This reduces stutter when input FPS
                # exceeds processing FPS and keeps the display closer to the latest frame.
                try:
                    self.input_queue.put(item, timeout=0.001)
                except queue.Full:
                    try:
                        # Drop one item first (remove the oldest frame)
                        dropped_item = self.input_queue.get_nowait()
                        # Increment the dropped-frame counter for the capture stage
                        with queue_drop_lock:
                            queue_drop_counts["input"][self.channel_id] += 1
                    except queue.Empty:
                        dropped_item = None

                    try:
                        # Then push the newest frame.
                        self.input_queue.put_nowait(item)
                    except queue.Full:
                        # In extreme cases, skip quietly without logging.
                        pass

                # Record read-stage throughput only when the frame was queued successfully
                record_throughput("read", time.time())

                # Apply a simple sleep if an FPS limit is configured
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
    engine: YOLOv11Engine,
    input_queue: "queue.Queue[CaptureItem]",
    infer_queue: "queue.Queue[InferItem]",
    stop_event: threading.Event,
) -> None:
    """Global preprocess + run_async worker.

    - Consumes frames from multiple channels through a single queue.
    - Runs preprocess, then run_async, and forwards req_id to infer_queue.
    """

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = time.perf_counter()
        input_tensor, meta_pre = engine.preprocess(item.frame_bgr)
        item.meta.update(meta_pre)
        item.meta["t_preprocess"] = time.perf_counter() - t0

        # Treat preprocess + run_async together as completion of the preprocess stage.
        req_id = engine.run_async(input_tensor)
        infer_item = InferItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            input_tensor=input_tensor,
            req_id=req_id,
            meta=item.meta,
        )

        # If infer_queue is full, drop the oldest item and enqueue the newest one.
        try:
            infer_queue.put(infer_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = infer_queue.get_nowait()
                # Increment the dropped-frame counter for the preprocess stage
                with queue_drop_lock:
                    queue_drop_counts["infer"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                infer_queue.put_nowait(infer_item)
            except queue.Full:
                # In extreme cases, skip quietly without logging.
                pass

        # Record preprocess-stage throughput when preprocess + run_async completes
        record_throughput("pre", time.time())


def wait_worker(
    engine: YOLOv11Engine,
    infer_queue: "queue.Queue[InferItem]",
    post_queue: "queue.Queue[OutputItem]",
    stop_event: threading.Event,
) -> None:
    """Global wait/inference worker.

    - Waits for req_id values from run_async, retrieves output_tensors, and forwards them to output_queue.
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

        # If post_queue is full, drop the oldest item and enqueue the newest one.
        try:
            post_queue.put(out_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = post_queue.get_nowait()
                # Increment the dropped-frame counter for the postprocess stage
                with queue_drop_lock:
                    queue_drop_counts["post"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                post_queue.put_nowait(out_item)
            except queue.Full:
                # In extreme cases, skip quietly without logging.
                pass

        # Record inference-stage throughput after wait completes and the item is queued for postprocess
        record_throughput("inf", time.time())


def postprocess_worker(
    engine: YOLOv11Engine,
    post_queue: "queue.Queue[OutputItem]",
    draw_queue: "queue.Queue[OutputItem]",
    # selected_classes is managed on the GUI side and exposed through a thread-safe reader
    get_selected_classes,
    stop_event: threading.Event,
) -> None:
    """Global postprocess + class-filter worker.

    - Receives result tensors and original frames from post_queue and postprocesses them.
    - Keeps only the selected classes and forwards the result to draw_queue.
    """

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = post_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        # engine.postprocess returns a (detections, masks) tuple
        detections, masks = engine.postprocess(item.output_tensors, item.meta)
        # detections = np.ones((1, 6))
        # masks = np.ones((1, 640, 640), dtype=np.uint8)
        t1 = time.perf_counter()

        # Read the currently selected class set and filter results
        selected_classes = get_selected_classes()

        if selected_classes is None:
            filtered_detections = detections
            filtered_masks = masks
        elif len(selected_classes) == 0:
            # If no class is selected, draw nothing.
            filtered_detections = np.empty((0, 6), dtype=detections.dtype)
            filtered_masks = (
                np.empty((0, *masks.shape[1:]), dtype=masks.dtype)
                if masks is not None and len(masks) > 0
                else np.empty((0, 0, 0), dtype=np.uint8)
            )
        else:
            cls_mask = np.isin(detections[:, 5].astype(int), list(selected_classes))
            filtered_detections = detections[cls_mask]
            filtered_masks = (
                masks[cls_mask] if masks is not None and len(masks) > 0 else masks
            )

        # Skip instances that are too small / low-value for visualisation to reduce draw cost.
        MIN_AREA_TO_DRAW = 20 * 20  # Assume areas smaller than 400px have little visual value

        if filtered_detections is not None and len(filtered_detections) > 0:
            keep_indices = []
            for idx, det in enumerate(filtered_detections):
                x1, y1, x2, y2, score, _ = det

                # When a mask exists, also enforce the minimum size using real mask pixels
                if (
                    filtered_masks is not None
                    and len(filtered_masks) > idx
                    and filtered_masks[idx] is not None
                ):
                    area = int(np.count_nonzero(filtered_masks[idx]))
                    if area < MIN_AREA_TO_DRAW:
                        continue
                else:
                    # If no mask exists, filter using bbox area only
                    bbox_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    if bbox_area < MIN_AREA_TO_DRAW:
                        continue

                keep_indices.append(idx)

            if keep_indices:
                keep_indices = np.array(keep_indices, dtype=np.int32)
                filtered_detections = filtered_detections[keep_indices]
                if filtered_masks is not None and len(filtered_masks) > 0:
                    filtered_masks = filtered_masks[keep_indices]
            else:
                # If everything was filtered out, replace with empty arrays
                filtered_detections = np.empty((0, 6), dtype=detections.dtype)
                if filtered_masks is not None and len(filtered_masks) > 0:
                    filtered_masks = np.empty(
                        (0, *filtered_masks.shape[1:]),
                        dtype=filtered_masks.dtype,
                    )

        t2 = time.perf_counter()

        # Record postprocess timings in meta (draw timing is recorded in draw_worker)
        item.meta["t_postprocess"] = t2 - t0
        item.meta["t_post_engine"] = t1 - t0
        item.meta["t_post_filter"] = t2 - t1
        item.meta["t_post_draw"] = 0.0

        # Store filtered results temporarily in meta and forward them to the draw stage
        item.meta["detections"] = filtered_detections
        item.meta["masks"] = filtered_masks

        try:
            draw_queue.put(item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = draw_queue.get_nowait()
                with queue_drop_lock:
                    queue_drop_counts["draw"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                draw_queue.put_nowait(item)
            except queue.Full:
                pass

        # Record postprocess-stage throughput after postprocess + filtering completes and the item is queued for draw
        record_throughput("post", time.time())


def draw_worker(
    engine: YOLOv11Engine,
    draw_queue: "queue.Queue[OutputItem]",
    on_frame_ready,
    stop_event: threading.Event,
) -> None:
    """Global draw + GUI forwarding worker.

    - Receives frame + meta(detections/masks) from draw_queue,
      calls draw_detections, and forwards the result via on_frame_ready.
    """

    last_log_ts = time.time()

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = draw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detections = item.meta.pop("detections", None)
        masks = item.meta.pop("masks", None)

        t_draw0 = time.perf_counter()
        engine.draw_detections(item.frame_bgr, detections, masks)
        t_draw1 = time.perf_counter()

        item.meta["t_post_draw"] = t_draw1 - t_draw0
        # Interpret total t_postprocess as post + draw combined
        item.meta["t_postprocess"] = (
            item.meta.get("t_post_engine", 0.0)
            + item.meta.get("t_post_filter", 0.0)
            + item.meta.get("t_post_draw", 0.0)
        )

        on_frame_ready(item.channel_id, item.frame_bgr, item.meta)

        # Record draw-stage throughput after visualisation and GUI forwarding complete
        now = time.time()
        record_throughput("draw", now)
