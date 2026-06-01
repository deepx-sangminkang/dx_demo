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


def test_build_video_pipeline_drains_audio_to_fakesink():
    # mp4/mov files carry an audio track; parsebin exposes it as a second pad.
    # Leaving it unlinked deadlocks parsebin's shared multiqueue (the decoder
    # starves and never prerolls), so the extra stream must be drained to a
    # fakesink. The video pad is routed by caps to the (video-only) decoder.
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
        rga_convert=True,
    )
    assert "parsebin name=" in pipeline
    assert "fakesink" in pipeline
    # The audio-drain sink must not block preroll when a file has no audio.
    assert "async=false" in pipeline
    # Exactly one named parsebin feeding two branches (video + audio drain).
    assert pipeline.count("parsebin") == 1
    # The decoder still receives the (video) stream.
    assert "mppvideodec" in pipeline


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


# ===== build_gst_pipeline: RGA dxconvert (RGB) path =====


def test_build_video_pipeline_rk3588_rga_outputs_rgb():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
        rga_convert=True,
    )
    assert "mppvideodec" in pipeline
    assert "dxconvert" in pipeline
    assert "format=RGB" in pipeline
    assert "format=BGR" not in pipeline


def test_build_rtsp_pipeline_rk3588_rga_outputs_rgb():
    pipeline = gp.build_gst_pipeline(
        source_type="rtsp",
        source="rtsp://10.0.0.1/s",
        platform=gp.Platform.RK3588,
        rga_convert=True,
    )
    assert "dxconvert" in pipeline
    assert "format=RGB" in pipeline


def test_build_camera_pipeline_rk3588_rga_stays_bgr():
    # Camera path keeps the CPU videoconvert->BGR chain even with rga_convert.
    pipeline = gp.build_gst_pipeline(
        source_type="camera",
        source=0,
        platform=gp.Platform.RK3588,
        rga_convert=True,
    )
    assert "dxconvert" not in pipeline
    assert "format=BGR" in pipeline


def test_build_video_pipeline_non_rk3588_ignores_rga():
    # dxconvert/RGA only applies on RK3588; other platforms stay BGR.
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.INTEL_VAAPI,
        rga_convert=True,
    )
    assert "dxconvert" not in pipeline
    assert "format=BGR" in pipeline


def test_hw_output_color_format():
    assert (
        gp.hw_output_color_format("video", gp.Platform.RK3588, True) == gp.COLOR_RGB
    )
    assert (
        gp.hw_output_color_format("rtsp", gp.Platform.RK3588, True) == gp.COLOR_RGB
    )
    assert (
        gp.hw_output_color_format("camera", gp.Platform.RK3588, True) == gp.COLOR_BGR
    )
    assert (
        gp.hw_output_color_format("video", gp.Platform.RK3588, False) == gp.COLOR_BGR
    )
    assert (
        gp.hw_output_color_format("video", gp.Platform.INTEL_VAAPI, True)
        == gp.COLOR_BGR
    )


# ===== build_gst_pipeline: RGA dxscale (HW resize offload) path =====


def test_build_video_pipeline_rga_scale_inserts_dxscale():
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
        rga_convert=True,
        scale_size=(640, 640),
    )
    assert "mppvideodec" in pipeline
    assert "dxscale" in pipeline
    assert "width=640" in pipeline
    assert "height=640" in pipeline
    assert "dxconvert" in pipeline
    # appsink caps pin the model input resolution so negotiation is explicit.
    assert "format=RGB,width=640,height=640" in pipeline
    # dxscale must run before the colour conversion (NV12 scale, then RGB).
    assert pipeline.index("dxscale") < pipeline.index("dxconvert")


def test_build_rtsp_pipeline_rga_scale_inserts_dxscale():
    pipeline = gp.build_gst_pipeline(
        source_type="rtsp",
        source="rtsp://10.0.0.1/s",
        platform=gp.Platform.RK3588,
        rga_convert=True,
        scale_size=(640, 480),
    )
    assert "dxscale" in pipeline
    assert "width=640" in pipeline
    assert "height=480" in pipeline
    assert "format=RGB,width=640,height=480" in pipeline


def test_build_video_pipeline_no_scale_when_size_none():
    # Default (scale_size=None) keeps the original behaviour: no dxscale.
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
        rga_convert=True,
    )
    assert "dxscale" not in pipeline
    assert "width=" not in pipeline


def test_build_video_pipeline_scale_ignored_without_rga():
    # dxscale relies on the RGB/dxconvert path; without rga_convert it is skipped.
    pipeline = gp.build_gst_pipeline(
        source_type="video",
        source="/data/a.mp4",
        platform=gp.Platform.RK3588,
        rga_convert=False,
        scale_size=(640, 640),
    )
    assert "dxscale" not in pipeline


def test_build_camera_pipeline_scale_ignored():
    # Camera keeps the CPU videoconvert chain; no dxscale even with scale_size.
    pipeline = gp.build_gst_pipeline(
        source_type="camera",
        source=0,
        platform=gp.Platform.RK3588,
        rga_convert=True,
        scale_size=(640, 640),
    )
    assert "dxscale" not in pipeline


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
    def __init__(self, opened: bool, grab_ok: bool = True):
        self._opened = opened
        self._grab_ok = grab_ok
        self.released = False

    def isOpened(self):
        return self._opened

    def grab(self):
        return self._grab_ok

    def release(self):
        self.released = True


def test_open_capture_hw_success(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        video_capture_factory=factory,
    )
    assert used_hw is True
    assert cap.isOpened()
    assert color_format == gp.COLOR_BGR
    # GStreamer pipeline string + CAP_GSTREAMER backend
    assert "mppvideodec" in calls[0][0]


def test_open_capture_hw_rga_reports_rgb(monkeypatch):
    def factory(arg, *rest):
        return _FakeCap(True)

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        rga_convert=True,
        video_capture_factory=factory,
    )
    assert used_hw is True
    assert color_format == gp.COLOR_RGB


def test_open_capture_hw_open_fails_falls_back_to_sw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        # First (HW) call fails, second (SW) call succeeds
        return _FakeCap(len(calls) >= 2)

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        rga_convert=True,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert cap.isOpened()
    assert len(calls) == 2
    # SW fallback is always BGR regardless of the requested RGA path.
    assert color_format == gp.COLOR_BGR
    # SW path opens the raw source
    assert calls[1][0] == "/data/a.mp4"


def test_open_capture_hw_opens_but_no_frames_falls_back_to_sw(monkeypatch):
    """A HW pipeline that reports isOpened() but yields no frames (caps never
    negotiated) must be released and fall back to SW decode."""

    caps = []

    def factory(arg, *rest):
        # First (HW) call: opens but grab() fails (no frames flow).
        # Second (SW) call: a normal working capture.
        if not caps:
            cap = _FakeCap(True, grab_ok=False)
        else:
            cap = _FakeCap(True, grab_ok=True)
        caps.append((cap, arg))
        return cap

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        rga_convert=True,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert cap.isOpened()
    assert color_format == gp.COLOR_BGR
    assert len(caps) == 2
    # The dead HW capture must be released before falling back.
    assert caps[0][0].released is True
    # SW fallback opens the raw source.
    assert caps[1][1] == "/data/a.mp4"


def test_open_capture_sw_mode_skips_hw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="sw",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=True,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert color_format == gp.COLOR_BGR
    assert len(calls) == 1
    assert calls[0][0] == "/data/a.mp4"


def test_open_capture_no_gstreamer_uses_sw(monkeypatch):
    calls = []

    def factory(arg, *rest):
        calls.append((arg, rest))
        return _FakeCap(True)

    cap, used_hw, reason, color_format = gp.open_capture(
        source="/data/a.mp4",
        source_type="video",
        decode_mode="auto",
        platform=gp.Platform.RK3588,
        opencv_gstreamer=False,
        video_capture_factory=factory,
    )
    assert used_hw is False
    assert color_format == gp.COLOR_BGR
    assert calls[0][0] == "/data/a.mp4"
