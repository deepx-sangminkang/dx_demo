"""Unit tests for the native dx_stream inference pipeline string builder.

Pure string construction, so these run without GStreamer / dx_stream present.
"""

from __future__ import annotations

import pytest

from demo import native_pipeline as npl


def test_missing_native_requirements_all_present():
    assert (
        npl.missing_native_requirements(
            element_available=lambda name: True, pydxs_available=True
        )
        == []
    )


def test_missing_native_requirements_reports_missing_element():
    missing = npl.missing_native_requirements(
        element_available=lambda name: name != "dxinfer",
        pydxs_available=True,
    )
    assert any("dxinfer" in m for m in missing)
    assert len(missing) == 1


def test_missing_native_requirements_reports_pydxs():
    missing = npl.missing_native_requirements(
        element_available=lambda name: True, pydxs_available=False
    )
    assert any("pydxs" in m for m in missing)


def test_missing_native_requirements_reports_all():
    missing = npl.missing_native_requirements(
        element_available=lambda name: False, pydxs_available=False
    )
    # 3 dx_stream elements + pydxs
    assert len(missing) == 4


def test_build_infer_pipeline_video_contains_core_elements():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/data/a.mp4",
        preprocess_cfg=npl.PreprocessCfg(
            preprocess_id=1, width=640, height=640, keep_ratio=True, pad_value=114
        ),
        infer_cfg=npl.InferCfg(inference_id=1, model_path="/m/yolo26n.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(
            inference_id=1,
            library_file_path="/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so",
            function_name="PostProcess",
        ),
        appsink_name="sink0",
    )
    assert "decodebin" in p
    assert "dxpreprocess" in p and "resize-width=640" in p and "resize-height=640" in p
    assert "dxinfer" in p and "model-path=/m/yolo26n.dxnn" in p
    assert "dxpostprocess" in p and "libpostprocess_yolo26od.so" in p
    assert "function-name=PostProcess" in p
    assert "appsink name=sink0" in p
    assert "drop=true" in p and "max-buffers=1" in p and "sync=false" in p
    # The metadata-bearing element (here dxpostprocess, since there is no
    # display scale) must be named so its src pad can be probed before the
    # videoconvert that drops the custom DXFrameMeta.
    meta_name = npl.meta_source_name("sink0")
    assert f"dxpostprocess" in p and f"name={meta_name}" in p
    assert p.index(f"name={meta_name}") < p.index("videoconvert")
    # The display branch must convert to RGB with explicit caps so the appsink
    # delivers deterministic 3-channel frames (dxpostprocess does not convert
    # pixels, so the raw frame would otherwise be the decoder's NV12/I420).
    assert "videoconvert" in p
    assert "video/x-raw,format=RGB" in p
    assert (
        p.index("dxpreprocess")
        < p.index("dxinfer")
        < p.index("dxpostprocess")
        < p.index("videoconvert")
        < p.index("appsink")
    )


def test_build_infer_pipeline_rtsp_and_display_scale():
    p = npl.build_infer_pipeline(
        source_type="rtsp",
        source="rtsp://10.0.0.1/s",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m/y.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s1",
        display_size=(320, 240),
    )
    assert "rtspsrc location=rtsp://10.0.0.1/s" in p
    assert "dxscale width=320 height=240" in p
    # Detections are probed on dxpostprocess (original coords); the display
    # dxscale is a pixels-only downscale placed strictly after the probe, so it
    # must NOT carry the meta name.
    meta_name = npl.meta_source_name("s1")
    assert f"dxpostprocess inference-id=1 library-file-path=/l.so " in p
    assert f"name={meta_name}" in p
    assert "dxscale width=320 height=240 name=" not in p
    assert (
        p.index(f"name={meta_name}")
        < p.index("dxscale")
        < p.index("videoconvert")
        < p.index("appsink")
    )


def test_build_infer_pipeline_osd_inserts_dxosd_after_scale_before_convert():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/v.mp4",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(
            library_file_path="/usr/local/share/gstdxstream/lib/libpostprocess_yolo26seg.so"
        ),
        appsink_name="s",
        display_size=(320, 240),
        osd=True,
    )
    # dxosd renders the segmentation overlay AFTER the display dxscale (so the
    # masks are blended on the small downscaled frame, not the full-res one) and
    # before the NV12->RGB convert (which drops the custom meta). dxosd rescales
    # draw coords to the live buffer size, so this stays correct.
    assert "dxosd" in p
    meta_name = npl.meta_source_name("s")
    assert (
        p.index(f"name={meta_name}")
        < p.index("dxscale")
        < p.index("dxosd")
        < p.index("videoconvert")
        < p.index("appsink")
    )


def test_build_infer_pipeline_osd_default_off():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/v.mp4",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s",
    )
    assert "dxosd" not in p


def test_build_infer_pipeline_rga_color_convert():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/v.mp4",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s",
        color_convert="rga",
    )
    # RGA path uses dxconvert (not the CPU videoconvert) for NV12->RGB.
    assert "dxconvert ! video/x-raw,format=RGB" in p
    assert "videoconvert" not in p


def test_build_infer_pipeline_invalid_color_convert_raises():
    with pytest.raises(ValueError):
        npl.build_infer_pipeline(
            source_type="video",
            source="/v.mp4",
            preprocess_cfg=npl.PreprocessCfg(),
            infer_cfg=npl.InferCfg(),
            postprocess_cfg=npl.PostprocessCfg(),
            appsink_name="s",
            color_convert="bogus",
        )


def test_build_infer_pipeline_sync_true_paces_to_native_fps():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/v.mp4",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s",
        sync=True,
    )
    assert "sync=true" in p
    assert "sync=false" not in p


def test_build_infer_pipeline_camera_integer_index():
    p = npl.build_infer_pipeline(
        source_type="camera",
        source=0,
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m/y.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s",
    )
    assert "v4l2src device=/dev/video0" in p


def test_build_infer_pipeline_keep_ratio_false_and_pad():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/v.mp4",
        preprocess_cfg=npl.PreprocessCfg(keep_ratio=False, pad_value=0),
        infer_cfg=npl.InferCfg(model_path="/m.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s",
    )
    assert "keep-ratio=false" in p
    assert "pad-value=0" in p


def test_build_infer_pipeline_invalid_source_raises():
    with pytest.raises(ValueError):
        npl.build_infer_pipeline(
            source_type="bogus",
            source="x",
            preprocess_cfg=npl.PreprocessCfg(),
            infer_cfg=npl.InferCfg(),
            postprocess_cfg=npl.PostprocessCfg(),
            appsink_name="s",
        )
