"""Unit tests for native-backend Qt signal payload builders."""

from __future__ import annotations

import numpy as np

from demo import native_signal as ns


def test_build_frame_payload_defaults():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    ch, out_frame, meta = ns.build_frame_payload(1, frame)
    assert ch == 1
    assert out_frame is frame
    assert meta["color_format"] == "rgb"
    assert isinstance(meta["ts"], float)


def test_build_frame_payload_color_format_override():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    _, _, meta = ns.build_frame_payload(0, frame, color_format="bgr")
    assert meta["color_format"] == "bgr"


def test_build_detection_payload_shape_and_meta():
    det = np.zeros((3, 6), dtype=np.float32)
    ch, out_det, meta = ns.build_detection_payload(2, det, t_inference=0.012)
    assert ch == 2
    assert out_det is det
    assert meta["t_inference"] == 0.012
    # The native pipeline does read/preprocess/infer/postprocess in GStreamer;
    # absent per-stage timings default to 0.0 so the metrics accumulator is safe.
    assert meta["t_read"] == 0.0
    assert meta["t_preprocess"] == 0.0
    assert meta["t_draw"] == 0.0


def test_build_detection_payload_no_timing():
    det = np.zeros((0, 6), dtype=np.float32)
    _, _, meta = ns.build_detection_payload(0, det)
    assert meta["t_inference"] == 0.0
