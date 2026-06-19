"""Unit tests for the pydxs bridge (import guard + graceful host fallback)."""

from __future__ import annotations

import numpy as np

from demo import pydxs_bridge as pb


class _Obj:
    def __init__(self, box, confidence, label):
        self.box = box
        self.confidence = confidence
        self.label = label


class _FakePydxs:
    """Stand-in for the native pydxs module."""

    def __init__(self, meta_by_addr):
        self._meta = meta_by_addr

    def dx_get_frame_meta(self, addr):
        return self._meta.get(addr)


def test_bridge_unavailable_without_pydxs():
    bridge = pb.PydxsBridge(pydxs_module=None)
    assert bridge.available is False
    # A buffer object still yields an empty detection array, never raises.
    det = bridge.detections_for_buffer(object())
    assert det.shape == (0, 6)


def test_bridge_reads_detections_via_hash():
    buf = object()
    addr = hash(buf)
    fake = _FakePydxs({addr: [_Obj([1, 2, 3, 4], 0.8, 7)]})
    bridge = pb.PydxsBridge(pydxs_module=fake)
    assert bridge.available is True
    det = bridge.detections_for_buffer(buf)
    assert det.shape == (1, 6)
    np.testing.assert_allclose(det[0], [1, 2, 3, 4, 0.8, 7], rtol=1e-6)


def test_bridge_no_meta_returns_empty():
    fake = _FakePydxs({})
    bridge = pb.PydxsBridge(pydxs_module=fake)
    det = bridge.detections_for_buffer(object())
    assert det.shape == (0, 6)


def test_bridge_tracks_meta_present_and_count_when_objects_exist():
    buf = object()
    addr = hash(buf)
    fake = _FakePydxs({addr: [_Obj([1, 2, 3, 4], 0.8, 7), _Obj([0, 0, 1, 1], 0.5, 2)]})
    bridge = pb.PydxsBridge(pydxs_module=fake)
    bridge.detections_for_buffer(buf)
    assert bridge.last_meta_present is True
    assert bridge.last_obj_count == 2


def test_bridge_tracks_meta_present_false_when_no_meta():
    fake = _FakePydxs({})  # dx_get_frame_meta returns None
    bridge = pb.PydxsBridge(pydxs_module=fake)
    bridge.detections_for_buffer(object())
    assert bridge.last_meta_present is False
    assert bridge.last_obj_count == 0


def test_bridge_tracks_meta_present_true_but_zero_objects():
    buf = object()
    addr = hash(buf)
    fake = _FakePydxs({addr: []})  # meta exists but no objects detected
    bridge = pb.PydxsBridge(pydxs_module=fake)
    bridge.detections_for_buffer(buf)
    assert bridge.last_meta_present is True
    assert bridge.last_obj_count == 0


def test_bridge_unavailable_sets_meta_present_none():
    bridge = pb.PydxsBridge(pydxs_module=None)
    bridge.detections_for_buffer(object())
    assert bridge.last_meta_present is None

