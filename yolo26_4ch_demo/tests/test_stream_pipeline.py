"""Unit tests for StreamPipeline (sample dispatch wiring).

The real GStreamer runtime / appsink is board-only, so we inject a fake ``gst``
module, a fake appsink sample, and a fake sample extractor to test that a new
sample drives both the frame callback and the detection callback correctly.
"""

from __future__ import annotations

import numpy as np

from demo import stream_pipeline as sp


class _FlowReturn:
    OK = "OK"


class _FakeGst:
    FlowReturn = _FlowReturn


class _FakeBridge:
    def __init__(self, det):
        self._det = det
        self.seen_buffer = None

    def detections_for_buffer(self, buf):
        self.seen_buffer = buf
        return self._det


def test_on_new_sample_dispatches_frame_and_detections():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    buffer = object()
    det = np.ones((2, 6), dtype=np.float32)
    bridge = _FakeBridge(det)

    frames, dets = [], []

    pipe = sp.StreamPipeline(
        channel_id=3,
        pipeline_str="fakesrc ! appsink name=s",
        bridge=bridge,
        frame_callback=lambda ch, f: frames.append((ch, f)),
        detection_callback=lambda ch, d: dets.append((ch, d)),
        gst=_FakeGst,
        sample_extractor=lambda sample: (frame, buffer),
    )

    ret = pipe._on_new_sample(appsink_with_sample=object())

    assert ret == _FlowReturn.OK
    assert frames == [(3, frame)]
    assert len(dets) == 1 and dets[0][0] == 3
    np.testing.assert_array_equal(dets[0][1], det)
    assert bridge.seen_buffer is buffer


def test_on_new_sample_no_frame_is_noop():
    bridge = _FakeBridge(np.zeros((0, 6), dtype=np.float32))
    frames, dets = [], []
    pipe = sp.StreamPipeline(
        channel_id=0,
        pipeline_str="x",
        bridge=bridge,
        frame_callback=lambda ch, f: frames.append((ch, f)),
        detection_callback=lambda ch, d: dets.append((ch, d)),
        gst=_FakeGst,
        sample_extractor=lambda sample: (None, None),
    )
    ret = pipe._on_new_sample(appsink_with_sample=object())
    assert ret == _FlowReturn.OK
    assert frames == [] and dets == []


def test_on_new_sample_swallows_callback_errors():
    bridge = _FakeBridge(np.zeros((1, 6), dtype=np.float32))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def boom(ch, f):
        raise RuntimeError("draw failed")

    pipe = sp.StreamPipeline(
        channel_id=1,
        pipeline_str="x",
        bridge=bridge,
        frame_callback=boom,
        detection_callback=lambda ch, d: None,
        gst=_FakeGst,
        sample_extractor=lambda sample: (frame, object()),
    )
    # Must not raise out of the GStreamer callback.
    ret = pipe._on_new_sample(appsink_with_sample=object())
    assert ret == _FlowReturn.OK
