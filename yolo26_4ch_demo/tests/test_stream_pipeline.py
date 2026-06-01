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


class _PtsBuffer:
    """Minimal stand-in for a GstBuffer exposing a PTS."""

    def __init__(self, pts):
        self.pts = pts


def test_buffer_pts_returns_none_for_missing_or_invalid():
    pipe = _make_pipe()
    assert pipe._buffer_pts(object()) is None
    # CLOCK_TIME_NONE-style sentinel is treated as "no usable PTS".
    pipe._gst.CLOCK_TIME_NONE = 18446744073709551615
    assert pipe._buffer_pts(_PtsBuffer(18446744073709551615)) is None
    assert pipe._buffer_pts(_PtsBuffer(1000)) == 1000


def test_stash_and_take_meta_roundtrip():
    pipe = _make_pipe()
    det = np.ones((3, 6), dtype=np.float32)
    pipe._stash_meta(500, det)
    taken = pipe._take_meta(500)
    np.testing.assert_array_equal(taken, det)
    # Once taken it is gone.
    assert pipe._take_meta(500) is None


def test_take_meta_drops_older_stale_entries():
    pipe = _make_pipe()
    pipe._stash_meta(10, "a")
    pipe._stash_meta(20, "b")
    pipe._stash_meta(30, "c")
    # Consuming PTS 20 must also evict the older PTS 10 (already-dropped frame).
    assert pipe._take_meta(20) == "b"
    assert pipe._take_meta(10) is None
    assert pipe._take_meta(30) == "c"


def test_stash_is_bounded():
    pipe = _make_pipe()
    pipe._META_STASH_MAX = 5
    for i in range(20):
        pipe._stash_meta(i, i)
    assert len(pipe._meta_stash) == 5
    # Oldest entries evicted; newest retained.
    assert pipe._take_meta(0) is None
    assert pipe._take_meta(19) == 19


def test_on_new_sample_prefers_pts_stash_over_appsink_buffer():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    appsink_buf = _PtsBuffer(700)
    bridge = _FakeBridge(np.zeros((0, 6), dtype=np.float32))
    stashed = np.ones((4, 6), dtype=np.float32)

    dets = []
    pipe = sp.StreamPipeline(
        channel_id=0,
        pipeline_str="x",
        bridge=bridge,
        frame_callback=lambda ch, f: None,
        detection_callback=lambda ch, d: dets.append(d),
        gst=_FakeGst,
        sample_extractor=lambda sample: (frame, appsink_buf),
    )
    # Meta captured on the pre-videoconvert pad, keyed by the same PTS.
    pipe._stash_meta(700, stashed)

    pipe._on_new_sample(appsink_with_sample=object())

    np.testing.assert_array_equal(dets[0], stashed)
    # The appsink buffer must NOT have been consulted when a stash hit exists.
    assert bridge.seen_buffer is None


def test_on_new_sample_falls_back_to_bridge_without_stash():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    appsink_buf = _PtsBuffer(700)
    det = np.ones((1, 6), dtype=np.float32)
    bridge = _FakeBridge(det)

    dets = []
    pipe = sp.StreamPipeline(
        channel_id=0,
        pipeline_str="x",
        bridge=bridge,
        frame_callback=lambda ch, f: None,
        detection_callback=lambda ch, d: dets.append(d),
        gst=_FakeGst,
        sample_extractor=lambda sample: (frame, appsink_buf),
    )

    pipe._on_new_sample(appsink_with_sample=object())

    np.testing.assert_array_equal(dets[0], det)
    assert bridge.seen_buffer is appsink_buf

class _SeekFlags:
    FLUSH = 1
    KEY_UNIT = 2
    SEGMENT = 4


class _SeekType:
    SET = "set"


class _Format:
    TIME = "time"


class _SeekGst:
    FlowReturn = _FlowReturn
    SeekFlags = _SeekFlags
    SeekType = _SeekType
    Format = _Format


class _FakePipeline:
    def __init__(self):
        self.seeks = []

    def seek(self, rate, fmt, flags, start_type, start, stop_type, stop):
        self.seeks.append((rate, fmt, flags, start_type, start, stop_type, stop))
        return True


def test_should_loop_true_when_enabled_with_pipeline():
    pipe = _make_pipe()
    pipe.loop = True
    pipe._pipeline = _FakePipeline()
    assert pipe._should_loop() is True


def test_should_loop_false_when_disabled():
    pipe = _make_pipe()
    pipe.loop = False
    pipe._pipeline = _FakePipeline()
    assert pipe._should_loop() is False


def test_should_loop_false_without_pipeline():
    pipe = _make_pipe()
    pipe.loop = True
    pipe._pipeline = None
    assert pipe._should_loop() is False


def test_should_arm_segment_loop_only_once():
    pipe = _make_pipe()
    pipe.loop = True
    pipe._pipeline = _FakePipeline()
    assert pipe._segment_armed is False
    assert pipe._should_arm_segment_loop() is True
    pipe._segment_armed = True
    assert pipe._should_arm_segment_loop() is False


def test_should_arm_segment_loop_false_when_not_looping():
    pipe = _make_pipe()
    pipe.loop = False
    pipe._pipeline = _FakePipeline()
    assert pipe._should_arm_segment_loop() is False


def test_segment_seek_flush_arms_segment_loop():
    pipe = _make_pipe()
    pipe._gst = _SeekGst
    pipe._pipeline = _FakePipeline()
    pipe._segment_seek(flush=True)
    assert pipe._pipeline.seeks == [
        (
            1.0,
            _Format.TIME,
            _SeekFlags.FLUSH | _SeekFlags.SEGMENT,
            _SeekType.SET,
            0,
            _SeekType.SET,
            -1,
        )
    ]


def test_segment_seek_non_flush_continues_loop():
    pipe = _make_pipe()
    pipe._gst = _SeekGst
    pipe._pipeline = _FakePipeline()
    pipe._segment_seek(flush=False)
    # The continuation seek must NOT flush (gapless), only SEGMENT.
    assert pipe._pipeline.seeks == [
        (
            1.0,
            _Format.TIME,
            _SeekFlags.SEGMENT,
            _SeekType.SET,
            0,
            _SeekType.SET,
            -1,
        )
    ]
