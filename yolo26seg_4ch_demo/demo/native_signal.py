"""Build Qt signal payloads for the native (dx_stream) backend.

The display path already consumes:
  * ``frame_ready(channel_id, frame, meta)``  with meta {color_format, ts}
  * ``detections_ready(channel_id, detections, meta)`` with meta carrying
    optional per-stage timings (t_read/t_preprocess/t_inference/t_draw).

These helpers produce identical payloads from native-pipeline samples so the
existing slots (``on_frame_ready`` / ``on_detections_ready``) need no changes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


def build_frame_payload(
    channel_id: int, frame: np.ndarray, color_format: str = "rgb"
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """Build a ``frame_ready`` payload. appsink frames are RGB by default."""
    meta = {"color_format": color_format, "ts": time.time()}
    return channel_id, frame, meta


def build_detection_payload(
    channel_id: int,
    detections: np.ndarray,
    t_inference: Optional[float] = None,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """Build a ``detections_ready`` payload with metrics-safe timing keys."""
    meta = {
        "t_read": 0.0,
        "t_preprocess": 0.0,
        "t_inference": float(t_inference) if t_inference is not None else 0.0,
        "t_draw": 0.0,
    }
    return channel_id, detections, meta
