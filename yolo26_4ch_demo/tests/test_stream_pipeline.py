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


def _make_pipe(channel_id=0, error_callback=None):
    return sp.StreamPipeline(
        channel_id=channel_id,
        pipeline_str="x",
        bridge=_FakeBridge(np.zeros((0, 6), dtype=np.float32)),
        frame_callback=lambda ch, f: None,
        detection_callback=lambda ch, d: None,
        gst=_FakeGst,
        sample_extractor=lambda sample: (None, None),
        error_callback=error_callback,
    )


def test_format_bus_error_includes_channel_source_and_debug():
    pipe = _make_pipe(channel_id=2)
    text = pipe._format_bus_error("dxinfer0", "could not load model", "gstdxinfer.c(120)")
    assert "Channel 2" in text
    assert "dxinfer0" in text
    assert "could not load model" in text
    assert "gstdxinfer.c(120)" in text


def test_dispatch_bus_error_invokes_error_callback():
    seen = []
    pipe = _make_pipe(channel_id=5, error_callback=lambda ch, msg: seen.append((ch, msg)))
    pipe._dispatch_bus_error("dxpostprocess0", "lib not found", None)
    assert len(seen) == 1
    assert seen[0][0] == 5
    assert "lib not found" in seen[0][1]


def test_dispatch_bus_error_without_callback_does_not_raise():
    pipe = _make_pipe(channel_id=0, error_callback=None)
    # Should log/print but never raise even with no callback wired.
    pipe._dispatch_bus_error("src", "boom", "dbg")


def test_should_log_sample_first_n_then_periodic():
    pipe = _make_pipe()
    assert pipe._should_log_sample(1) is True
    assert pipe._should_log_sample(3) is True
    assert pipe._should_log_sample(4) is False
    assert pipe._should_log_sample(300) is True
    assert pipe._should_log_sample(301) is False


def test_on_new_sample_logs_detection_diagnostics(capsys):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    bridge = _FakeBridge(np.ones((2, 6), dtype=np.float32))
    bridge.last_meta_present = True
    pipe = sp.StreamPipeline(
        channel_id=1,
        pipeline_str="x",
        bridge=bridge,
        frame_callback=lambda ch, f: None,
        detection_callback=lambda ch, d: None,
        gst=_FakeGst,
        sample_extractor=lambda sample: (frame, object()),
    )
    pipe._on_new_sample(appsink_with_sample=object())
    err = capsys.readouterr().err
    assert "DXS-DEBUG" in err
    assert "detections=2" in err
    assert "frame_meta_present=True" in err

