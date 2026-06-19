"""Adapt pydxs ``DXFrameMeta`` into the demo's detection ndarray contract.

The display path consumes detections as an ``(N, 6)`` float32 array of
``[x1, y1, x2, y2, score, class_id]`` in original-frame coordinates. dxpostprocess
already produces original-frame boxes (letterbox removed), so this is a thin map.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np

_EMPTY = np.zeros((0, 6), dtype=np.float32)


def frame_meta_to_detections(frame_meta: Optional[Iterable]) -> np.ndarray:
    """Convert an iterable of DXObjectMeta into an ``(N, 6)`` float32 array."""
    if frame_meta is None:
        return _EMPTY.copy()

    rows = []
    for obj in frame_meta:
        box = obj.box
        rows.append(
            [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(obj.confidence),
                float(obj.label),
            ]
        )

    if not rows:
        return _EMPTY.copy()
    return np.asarray(rows, dtype=np.float32)


def filter_by_classes(
    detections: np.ndarray, selected: Optional[Set[int]]
) -> np.ndarray:
    """Keep only rows whose class_id is in ``selected`` (``None`` keeps all)."""
    if selected is None or detections.shape[0] == 0:
        return detections
    mask = np.isin(detections[:, 5].astype(np.int64), list(selected))
    return detections[mask]
