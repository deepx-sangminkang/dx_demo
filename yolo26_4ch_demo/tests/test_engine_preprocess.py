"""Unit tests for YOLO26Engine.preprocess colour handling.

Verifies that the RGB end-to-end path (RGA dxconvert HW decode) skips the
BGR->RGB conversion, while the default BGR path still converts.
"""

from __future__ import annotations

import numpy as np

from demo.engine import YOLO26Engine


def _make_engine(input_size: int = 64) -> YOLO26Engine:
    """Build an engine instance without running the (model-loading) __init__."""

    engine = YOLO26Engine.__new__(YOLO26Engine)
    engine.input_height = input_size
    engine.input_width = input_size
    return engine


def test_preprocess_bgr_converts_to_rgb():
    engine = _make_engine()
    # Distinct channels so a BGR<->RGB swap is observable.
    frame_bgr = np.zeros((32, 48, 3), dtype=np.uint8)
    frame_bgr[..., 0] = 10  # B
    frame_bgr[..., 1] = 20  # G
    frame_bgr[..., 2] = 30  # R

    tensor, meta = engine.preprocess(frame_bgr, color_format="bgr")

    # After BGR->RGB, channel 0 should carry the original R (=30) where unpadded.
    # Sample the centre pixel (inside the resized content, not the gray pad).
    cy, cx = tensor.shape[0] // 2, tensor.shape[1] // 2
    assert tensor[cy, cx, 0] == 30
    assert tensor[cy, cx, 2] == 10
    assert meta["orig_shape"] == (32, 48)


def test_preprocess_rgb_skips_conversion():
    engine = _make_engine()
    frame_rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    frame_rgb[..., 0] = 30  # already R
    frame_rgb[..., 1] = 20  # G
    frame_rgb[..., 2] = 10  # B

    tensor, _ = engine.preprocess(frame_rgb, color_format="rgb")

    cy, cx = tensor.shape[0] // 2, tensor.shape[1] // 2
    # No channel swap: channel 0 stays 30, channel 2 stays 10.
    assert tensor[cy, cx, 0] == 30
    assert tensor[cy, cx, 2] == 10
