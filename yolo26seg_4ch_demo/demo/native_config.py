"""Resolve the native (dx_stream) backend configuration from the demo config.

Keeps config parsing pure and unit-testable, separate from Qt wiring.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .native_pipeline import InferCfg, PostprocessCfg, PreprocessCfg

_VALID_BACKENDS = ("dxstream",)


def get_engine_backend(config: Dict[str, Any]) -> str:
    """Return the selected inference backend.

    The demo is dx_stream-only; ``dxstream`` is the sole valid value and the
    default. The key is still validated so a stale ``legacy`` config fails
    loudly instead of silently doing nothing.
    """
    backend = str(config.get("engine_backend", "dxstream"))
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"engine_backend must be one of {_VALID_BACKENDS}, got {backend!r}. "
            "The legacy OpenCV backend has been removed; use 'dxstream'."
        )
    return backend


def build_native_cfgs(
    config: Dict[str, Any], input_width: int, input_height: int
) -> Tuple[PreprocessCfg, InferCfg, PostprocessCfg]:
    """Build dx_stream element configs from the demo config + model input size."""
    model_path = config.get("model")
    if not model_path:
        raise ValueError("config 'model' is required for dxstream backend")

    dxs = config.get("dxstream") or {}
    library = dxs.get("postprocess_library")
    if not library:
        raise ValueError(
            "config dxstream.postprocess_library is required for dxstream backend"
        )

    ident = int(dxs.get("preprocess_id", 1))
    pre = PreprocessCfg(
        preprocess_id=ident,
        width=int(input_width),
        height=int(input_height),
        keep_ratio=bool(dxs.get("keep_ratio", True)),
        pad_value=int(dxs.get("pad_value", 114)),
    )
    inf = InferCfg(
        inference_id=ident,
        model_path=str(model_path),
        use_ort=bool(dxs.get("use_ort", False)),
    )
    post = PostprocessCfg(
        inference_id=ident,
        library_file_path=str(library),
        function_name=str(dxs.get("postprocess_function", "PostProcess")),
    )
    return pre, inf, post
