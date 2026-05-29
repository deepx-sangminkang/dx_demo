"""Unit tests for YOLO26Engine.finalize_detections.

Verifies class filtering, score thresholding, and letterbox-inverse coordinate
conversion used by the display-inference decoupling path.
"""

from __future__ import annotations

import numpy as np

from demo.engine import YOLO26Engine


def _make_engine() -> YOLO26Engine:
    """Build an engine instance without running the (model-loading) __init__."""

    engine = YOLO26Engine.__new__(YOLO26Engine)
    engine.score_threshold = 0.4
    return engine


def _meta(orig=(100, 200)):
    # No padding, scale 1.0 -> coords pass through (after clipping).
    return {
        "orig_shape": orig,
        "pad_top": 0,
        "pad_left": 0,
        "scale": 1.0,
    }


def test_finalize_filters_by_score_threshold():
    engine = _make_engine()
    # Two boxes: one above, one below threshold.
    dets = np.array(
        [
            [10, 10, 20, 20, 0.9, 0],
            [30, 30, 40, 40, 0.1, 0],
        ],
        dtype=np.float32,
    )
    out = engine.finalize_detections([dets], _meta())
    assert len(out) == 1
    assert out[0, 4] == np.float32(0.9)


def test_finalize_filters_by_selected_classes():
    engine = _make_engine()
    dets = np.array(
        [
            [10, 10, 20, 20, 0.9, 0],
            [30, 30, 40, 40, 0.9, 5],
        ],
        dtype=np.float32,
    )
    out = engine.finalize_detections([dets], _meta(), selected_classes={5})
    assert len(out) == 1
    assert int(out[0, 5]) == 5


def test_finalize_empty_selection_returns_empty():
    engine = _make_engine()
    dets = np.array([[10, 10, 20, 20, 0.9, 0]], dtype=np.float32)
    out = engine.finalize_detections([dets], _meta(), selected_classes=set())
    assert len(out) == 0


def test_finalize_clips_to_original_bounds():
    engine = _make_engine()
    # Box extends beyond original 100x200 image; should be clipped.
    dets = np.array([[-5, -5, 999, 999, 0.9, 0]], dtype=np.float32)
    out = engine.finalize_detections([dets], _meta(orig=(100, 200)))
    assert out[0, 0] >= 0 and out[0, 1] >= 0
    assert out[0, 2] <= 199 and out[0, 3] <= 99
