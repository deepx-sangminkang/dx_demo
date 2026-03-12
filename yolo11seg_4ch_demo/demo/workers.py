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


def _increment_drop_count(stage: str, channel_id: Optional[int]) -> None:
    """Increment the dropped-item counter for a queue stage."""

    if channel_id is None:
        return

    with queue_drop_lock:
        queue_drop_counts[stage][channel_id] += 1


def _empty_masks_like(masks: Optional[np.ndarray]) -> np.ndarray:
    """Create an empty mask array compatible with the current mask layout."""

    if masks is not None and len(masks) > 0:
        return np.empty((0, *masks.shape[1:]), dtype=masks.dtype)
    return np.empty((0, 0, 0), dtype=np.uint8)


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


def _filter_selected_detections(
    detections: np.ndarray,
    masks: Optional[np.ndarray],
    selected_classes: Optional[set[int]],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Filter model outputs by the currently selected classes."""

    if selected_classes is None:
        return detections, masks

    if len(selected_classes) == 0:
        return np.empty((0, 6), dtype=detections.dtype), _empty_masks_like(masks)

    cls_mask = np.isin(detections[:, 5].astype(int), list(selected_classes))
    filtered_detections = detections[cls_mask]
    filtered_masks = masks[cls_mask] if masks is not None and len(masks) > 0 else masks
    return filtered_detections, filtered_masks


def _meets_min_area(det: np.ndarray, masks: Optional[np.ndarray], idx: int) -> bool:
    """Return whether a detection is large enough to keep for drawing."""

    min_area_to_draw = 20 * 20
    x1, y1, x2, y2, _score, _class_id = det

    if masks is not None and len(masks) > idx and masks[idx] is not None:
        return int(np.count_nonzero(masks[idx])) >= min_area_to_draw

    bbox_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return bbox_area >= min_area_to_draw


def _filter_small_detections(
    detections: np.ndarray,
    masks: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop detections that are too small to be useful on screen."""

    if detections is None or len(detections) == 0:
        return detections, masks

    keep_indices = [idx for idx, det in enumerate(detections) if _meets_min_area(det, masks, idx)]
    if not keep_indices:
        return np.empty((0, 6), dtype=detections.dtype), _empty_masks_like(masks)

    keep_array = np.array(keep_indices, dtype=np.int32)
    kept_masks = masks[keep_array] if masks is not None and len(masks) > 0 else masks
    return detections[keep_array], kept_masks


def _update_postprocess_meta(
    meta: Dict[str, Any],
    detections: np.ndarray,
    masks: Optional[np.ndarray],
    start_ts: float,
    engine_ts: float,
    end_ts: float,
) -> None:
    """Store postprocess timing and filtered outputs in frame metadata."""

    meta["t_postprocess"] = end_ts - start_ts
    meta["t_post_engine"] = engine_ts - start_ts
    meta["t_post_filter"] = end_ts - engine_ts
    meta["t_post_draw"] = 0.0
    meta["detections"] = detections
    meta["masks"] = masks


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

    def _read_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Read one frame, rewinding file sources when EOF is reached."""

        ok, frame_bgr = cap.read()
        if ok:
            return frame_bgr

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame_bgr = cap.read()
        if ok:
            return frame_bgr

        print(f"[INFO] Channel {self.channel_id}: no more frames can be read (EOF or error)")
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
        """Sleep just enough to respect an optional FPS limit."""

        if min_interval <= 0.0:
            return

        elapsed = time.perf_counter() - start_ts
        remain = min_interval - elapsed
        if remain > 0:
            time.sleep(remain)

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
                frame_bgr = self._read_frame(cap)
                if frame_bgr is None:
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

                self._enqueue_capture_item(item)

                # Record read-stage throughput only when the frame was queued successfully
                record_throughput("read", time.time())

                self._sleep_for_fps_limit(t0, min_interval)
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

        if item is None:
            break

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

        _enqueue_latest(infer_queue, infer_item, stage="infer")

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

        if item is None:
            break

        t0 = time.perf_counter()
        output_tensors = engine.wait(item.req_id)
        item.meta["t_inference"] = time.perf_counter() - t0

        out_item = OutputItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            output_tensors=output_tensors,
            meta=item.meta,
        )

        _enqueue_latest(post_queue, out_item, stage="post")

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

        if item is None:
            break

        t0 = time.perf_counter()
        detections, masks = engine.postprocess(item.output_tensors, item.meta)
        t1 = time.perf_counter()

        selected_classes = get_selected_classes()

        filtered_detections, filtered_masks = _filter_selected_detections(
            detections,
            masks,
            selected_classes,
        )
        filtered_detections, filtered_masks = _filter_small_detections(
            filtered_detections,
            filtered_masks,
        )

        t2 = time.perf_counter()

        _update_postprocess_meta(item.meta, filtered_detections, filtered_masks, t0, t1, t2)
        _enqueue_latest(draw_queue, item, stage="draw")

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

    while not stop_event.is_set():  # pragma: no cover - runtime only
        try:
            item = draw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

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
