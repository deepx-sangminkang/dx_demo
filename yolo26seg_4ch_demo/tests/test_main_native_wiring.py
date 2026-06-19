"""Host-side unit tests for the dxstream backend Qt glue in MainWindow.

We avoid building the full GUI; instead we bind the unbound methods to a tiny
stub that provides exactly the collaborators the glue touches. This verifies the
GStreamer-thread callbacks build correct payloads and apply the class filter.
"""

from __future__ import annotations

import types

import numpy as np

from demo.main import MainWindow


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


def _make_stub(selected):
    stub = types.SimpleNamespace()
    stub.frame_ready = _Signal()
    stub.detections_ready = _Signal()
    stub.get_selected_classes = lambda: selected
    return stub


def test_on_native_frame_emits_rgb_payload():
    stub = _make_stub({0})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    MainWindow._on_native_frame(stub, 2, frame)
    assert len(stub.frame_ready.emitted) == 1
    ch, out_frame, meta = stub.frame_ready.emitted[0]
    assert ch == 2 and out_frame is frame
    assert meta["color_format"] == "rgb"


def test_on_native_detections_filters_and_emits():
    stub = _make_stub({5})
    det = np.array(
        [
            [0, 0, 1, 1, 0.9, 0],
            [0, 0, 1, 1, 0.8, 5],
        ],
        dtype=np.float32,
    )
    MainWindow._on_native_detections(stub, 1, det)
    assert len(stub.detections_ready.emitted) == 1
    ch, out_det, meta = stub.detections_ready.emitted[0]
    assert ch == 1
    assert out_det.shape == (1, 6)
    assert out_det[0, 5] == 5.0
    assert "t_inference" in meta


def test_on_native_detections_none_filter_keeps_all():
    stub = _make_stub(None)
    det = np.zeros((3, 6), dtype=np.float32)
    MainWindow._on_native_detections(stub, 0, det)
    _, out_det, _ = stub.detections_ready.emitted[0]
    assert out_det.shape == (3, 6)
