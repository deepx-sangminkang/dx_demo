"""Unit tests for native (dx_stream) backend config resolution."""

from __future__ import annotations

import pytest

from demo import native_config as nc
from demo import native_pipeline as npl


def test_default_backend_is_dxstream():
    assert nc.get_engine_backend({}) == "dxstream"


def test_backend_dxstream_selected():
    assert nc.get_engine_backend({"engine_backend": "dxstream"}) == "dxstream"


def test_backend_invalid_raises():
    with pytest.raises(ValueError):
        nc.get_engine_backend({"engine_backend": "wat"})


def test_legacy_backend_rejected():
    with pytest.raises(ValueError):
        nc.get_engine_backend({"engine_backend": "legacy"})


def test_build_native_cfgs_defaults():
    cfg = {
        "model": "assets/models/yolo26n-1.dxnn",
        "dxstream": {
            "postprocess_library": "/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so",
        },
    }
    pre, inf, post = nc.build_native_cfgs(cfg, input_width=640, input_height=640)
    assert isinstance(pre, npl.PreprocessCfg)
    assert pre.width == 640 and pre.height == 640 and pre.keep_ratio is True
    assert inf.model_path == "assets/models/yolo26n-1.dxnn"
    assert post.library_file_path.endswith("libpostprocess_yolo26od.so")
    assert post.function_name == "PostProcess"
    assert pre.preprocess_id == inf.inference_id == post.inference_id


def test_build_native_cfgs_overrides():
    cfg = {
        "model": "/m/custom.dxnn",
        "dxstream": {
            "postprocess_library": "/x/lib.so",
            "postprocess_function": "MyPost",
            "keep_ratio": False,
            "pad_value": 0,
        },
    }
    pre, inf, post = nc.build_native_cfgs(cfg, input_width=512, input_height=512)
    assert pre.width == 512 and pre.keep_ratio is False and pre.pad_value == 0
    assert inf.model_path == "/m/custom.dxnn"
    assert post.library_file_path == "/x/lib.so"
    assert post.function_name == "MyPost"


def test_build_native_cfgs_requires_postprocess_library():
    with pytest.raises(ValueError):
        nc.build_native_cfgs({"model": "/m.dxnn"}, input_width=640, input_height=640)
