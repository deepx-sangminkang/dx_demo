"""Unit tests for the GStreamer HW decoding pipeline builder.

These tests cover pure logic (platform detection, pipeline string
construction, OpenCV GStreamer capability detection, and SW fallback
decisions) so they run without any real hardware decoder.
"""

from __future__ import annotations

import pytest

from demo import gst_pipeline as gp


# ===== opencv_has_gstreamer =====


def test_opencv_has_gstreamer_true():
    build_info = "  Video I/O:\n    GStreamer:                   YES (1.20.3)\n    FFMPEG:                      YES\n"
    assert gp.opencv_has_gstreamer(build_info) is True


def test_opencv_has_gstreamer_false():
    build_info = "  Video I/O:\n    GStreamer:                   NO\n    FFMPEG:                      YES\n"
    assert gp.opencv_has_gstreamer(build_info) is False


def test_opencv_has_gstreamer_missing_line():
    build_info = "  Video I/O:\n    FFMPEG:                      YES\n"
    assert gp.opencv_has_gstreamer(build_info) is False


# ===== detect_platform =====


def test_detect_platform_rk3588():
    probe = gp.PlatformProbe(
        device_tree_compatible="rockchip,rk3588-orangepi-5-plus\x00rockchip,rk3588\x00",
        available_elements=set(),
        dri_render_nodes=[],
    )
    assert gp.detect_platform(probe) == gp.Platform.RK3588


def test_detect_platform_nvidia():
    probe = gp.PlatformProbe(
        device_tree_compatible="",
        available_elements={"nvh264dec"},
        dri_render_nodes=[],
    )
    assert gp.detect_platform(probe) == gp.Platform.NVIDIA


def test_detect_platform_nvidia_v4l2():
    probe = gp.PlatformProbe(
        device_tree_compatible="",
        available_elements={"nvv4l2decoder"},
        dri_render_nodes=[],
    )
    assert gp.detect_platform(probe) == gp.Platform.NVIDIA


def test_detect_platform_intel_vaapi():
    probe = gp.PlatformProbe(
        device_tree_compatible="",
        available_elements={"vaapidecodebin", "vapostproc"},
        dri_render_nodes=["/dev/dri/renderD128"],
    )
    assert gp.detect_platform(probe) == gp.Platform.INTEL_VAAPI


def test_detect_platform_intel_requires_render_node():
    # VAAPI element exists but no render node -> cannot use Intel HW path
    probe = gp.PlatformProbe(
        device_tree_compatible="",
        available_elements={"vaapidecodebin"},
        dri_render_nodes=[],
    )
    assert gp.detect_platform(probe) == gp.Platform.UNKNOWN


def test_detect_platform_unknown():
    probe = gp.PlatformProbe(
        device_tree_compatible="",
        available_elements=set(),
        dri_render_nodes=[],
    )
    assert gp.detect_platform(probe) == gp.Platform.UNKNOWN


# ===== build_gst_pipeline: video file =====


def test_build_video_pipeline_rk3588():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
    )
    assert "filesrc location=/data/a.mp4" in pipeline
    assert "mppvideodec" in pipeline
    assert "format=BGR" in pipeline
    assert pipeline.strip().endswith("appsink") or "appsink" in pipeline


def test_build_video_pipeline_intel():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.INTEL_VAAPI,
    )
    assert "filesrc location=/data/a.mp4" in pipeline
    assert "vaapi" in pipeline or "vah264" in pipeline or "va" in pipeline
    assert "format=BGR" in pipeline
    assert "appsink" in pipeline


def test_build_video_pipeline_nvidia():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.NVIDIA,
    )
    assert "filesrc location=/data/a.mp4" in pipeline
    assert "dec" in pipeline
    assert "format=BGR" in pipeline


# ===== build_gst_pipeline: rtsp =====


def test_build_rtsp_pipeline():
    url = "rtsp://user:pass@10.0.0.1:554/stream"
    pipeline = gp.build_gst_pipeline(
        source_type="rtsp",
        source=url,
        platform=gp.Platform.RK3588,
    )
    assert f"rtspsrc location={url}" in pipeline
    assert "depay" in pipeline
    assert "mppvideodec" in pipeline
    assert "format=BGR" in pipeline
    assert "appsink" in pipeline


# ===== build_gst_pipeline: camera =====


def test_build_camera_pipeline_integer_index():
    pipeline = gp.build_gst_pipeline(
        source_type="camera",
        source=0,
        platform=gp.Platform.INTEL_VAAPI,
    )
    assert "v4l2src device=/dev/video0" in pipeline
    assert "format=BGR" in pipeline
    assert "appsink" in pipeline


def test_build_camera_pipeline_device_path():
    pipeline = gp.build_gst_pipeline(
        source_type="camera",
        source="/dev/video2",
        platform=gp.Platform.INTEL_VAAPI,
    )
    assert "v4l2src device=/dev/video2" in pipeline


# ===== build_gst_pipeline: appsink properties =====


def test_pipeline_appsink_drops_old_buffers():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
    )
    assert "drop=true" in pipeline
    assert "max-buffers=1" in pipeline


# ===== build_gst_pipeline: errors =====


def test_build_pipeline_unknown_platform_raises():
    with pytest.raises(gp.HwDecodeUnavailable):
        gp.build_gst_pipeline(
            source_type="video",
            source="/data/a.mp4",
            platform=gp.Platform.UNKNOWN,
        )


def test_build_pipeline_unsupported_type_raises():
    with pytest.raises(ValueError):
        gp.build_gst_pipeline(
            source_type="bogus",
            source="x",
            platform=gp.Platform.RK3588,
        )


# ===== should_use_hw_decode (fallback decision) =====


def test_should_use_hw_decode_sw_mode():
    decision = gp.resolve_decode(
        decode_mode="sw",
        opencv_gstreamer=True,
        platform=gp.Platform.RK3588,
    )
    assert decision.use_hw is False
    assert decision.reason  # has explanation


def test_should_use_hw_decode_hw_mode_ok():
    decision = gp.resolve_decode(
        decode_mode="hw",
        opencv_gstreamer=True,
        platform=gp.Platform.RK3588,
    )
    assert decision.use_hw is True


def test_should_use_hw_decode_auto_no_gstreamer_falls_back():
    decision = gp.resolve_decode(
        decode_mode="auto",
        opencv_gstreamer=False,
        platform=gp.Platform.RK3588,
    )
    assert decision.use_hw is False


def test_should_use_hw_decode_auto_unknown_platform_falls_back():
    decision = gp.resolve_decode(
        decode_mode="auto",
        opencv_gstreamer=True,
        platform=gp.Platform.UNKNOWN,
    )
    assert decision.use_hw is False


def test_should_use_hw_decode_hw_mode_no_gstreamer_falls_back():
    decision = gp.resolve_decode(
        decode_mode="hw",
        opencv_gstreamer=False,
        platform=gp.Platform.RK3588,
    )
    assert decision.use_hw is False


# ===== open_capture: HW open with SW fallback =====


class _FakeCap:
    def __init__(self, opened: bool):
        self._opened = opened

    def isOpened(self):
        return self._opened


def test_open_capture_hw_success(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        video_capture_factory=factory,
    )
    assert used_hw is True
    assert cap.isOpened()
    # GStreamer pipeline string + CAP_GSTREAMER backend
    assert "mppvideodec" in calls[0][0]


def test_open_capture_hw_open_fails_falls_back_to_sw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        # First (HW) call fails, second (SW) call succeeds
        return _FakeCap(len(calls) >= 2)

    cap, used_hw, reason = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert cap.isOpened()
    assert len(calls) == 2
    # SW path opens the raw source
    assert calls[1][0] == "/data/a.mp4"


def test_open_capture_sw_mode_skips_hw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="sw",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert len(calls) == 1
    assert calls[0][0] == "/data/a.mp4"


def test_open_capture_no_gstreamer_uses_sw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=False,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert calls[0][0] == "/data/a.mp4"
