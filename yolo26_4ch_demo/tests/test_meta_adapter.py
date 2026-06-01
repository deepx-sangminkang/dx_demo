"""Unit tests for converting DXFrameMeta -> demo detection format.

The demo's display path expects detections as an ``(N, 6)`` float32 ndarray of
``[x1, y1, x2, y2, score, class_id]`` in *original-frame* coordinates. dxpostprocess
already emits boxes in original-frame coordinates, so this adapter is a thin map.

A tiny fake frame-meta (iterable of objects) stands in for pydxs DXFrameMeta.
"""

from __future__ import annotations

import numpy as np

from demo import meta_adapter as ma


class _Obj:
    def __init__(self, box, confidence, label):
        self.box = box
        self.confidence = confidence
        self.label = label


class _FrameMeta:
    def __init__(self, objs):
        self._objs = objs

    def __iter__(self):
        return iter(self._objs)


def test_frame_meta_to_detections_basic_shape_and_values():
    fm = _FrameMeta([
        _Obj([10.0, 20.0, 110.0, 220.0], 0.9, 0),
        _Obj([5.0, 6.0, 7.0, 8.0], 0.5, 2),
    ])
    det = ma.frame_meta_to_detections(fm)
    assert det.shape == (2, 6)
    assert det.dtype == np.float32
    np.testing.assert_allclose(det[0], [10, 20, 110, 220, 0.9, 0], rtol=1e-6)
    np.testing.assert_allclose(det[1], [5, 6, 7, 8, 0.5, 2], rtol=1e-6)


def test_frame_meta_to_detections_empty():
    det = ma.frame_meta_to_detections(_FrameMeta([]))
    assert det.shape == (0, 6)
    assert det.dtype == np.float32


def test_frame_meta_to_detections_none():
    det = ma.frame_meta_to_detections(None)
    assert det.shape == (0, 6)


def test_filter_by_classes_keeps_selected():
    det = np.array(
        [
            [0, 0, 1, 1, 0.9, 0],
            [0, 0, 1, 1, 0.8, 2],
            [0, 0, 1, 1, 0.7, 5],
        ],
        dtype=np.float32,
    )
    out = ma.filter_by_classes(det, {0, 5})
    assert out.shape == (2, 6)
    assert set(out[:, 5].tolist()) == {0.0, 5.0}


def test_filter_by_classes_none_means_all():
    det = np.array([[0, 0, 1, 1, 0.9, 0]], dtype=np.float32)
    assert ma.filter_by_classes(det, None).shape == (1, 6)


def test_filter_by_classes_empty_input():
    det = np.zeros((0, 6), dtype=np.float32)
    assert ma.filter_by_classes(det, {1}).shape == (0, 6)
