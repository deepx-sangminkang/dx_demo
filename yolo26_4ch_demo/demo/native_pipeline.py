"""Native dx_stream inference pipeline string builder.

Builds GStreamer launch strings that run the full preprocess->infer->postprocess
chain on dx_stream HW elements (``dxpreprocess`` / ``dxinfer`` / ``dxpostprocess``),
terminating in an ``appsink`` so a Python/Qt front end can read decoded frames
and detection metadata (via ``pydxs``).

Pure string construction so it is unit-testable without GStreamer present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Keep only the newest decoded sample, never block, ignore the clock, and
# emit ``new-sample`` so the Python side can pull frames + metadata.
_APPSINK_OPTS = "drop=true max-buffers=1 sync=false emit-signals=true"


@dataclass
class PreprocessCfg:
    """dxpreprocess (RGA letterbox/resize) settings."""

    preprocess_id: int = 1
    width: int = 640
    height: int = 640
    keep_ratio: bool = True
    pad_value: int = 114


@dataclass
class InferCfg:
    """dxinfer (NPU) settings."""

    inference_id: int = 1
    model_path: str = ""


@dataclass
class PostprocessCfg:
    """dxpostprocess (decoder library) settings."""

    inference_id: int = 1
    library_file_path: str = ""
    function_name: str = "PostProcess"


def _preprocess_element(cfg: PreprocessCfg) -> str:
    return (
        f"dxpreprocess preprocess-id={cfg.preprocess_id} "
        f"resize-width={cfg.width} resize-height={cfg.height} "
        f"keep-ratio={'true' if cfg.keep_ratio else 'false'} "
        f"pad-value={cfg.pad_value}"
    )


def _infer_element(cfg: InferCfg, preprocess_id: int) -> str:
    return (
        f"dxinfer preprocess-id={preprocess_id} "
        f"inference-id={cfg.inference_id} model-path={cfg.model_path}"
    )


def _postprocess_element(cfg: PostprocessCfg) -> str:
    return (
        f"dxpostprocess inference-id={cfg.inference_id} "
        f"library-file-path={cfg.library_file_path} "
        f"function-name={cfg.function_name}"
    )


def _source_chain(source_type: str, source: Union[int, str]) -> str:
    if source_type == "video":
        return f"filesrc location={source} ! decodebin"
    if source_type == "rtsp":
        return (
            f"rtspsrc location={source} latency=100 ! "
            f"rtph264depay ! h264parse ! decodebin"
        )
    if source_type == "camera":
        text = str(source)
        dev = text if text.startswith("/dev/") else f"/dev/video{text}"
        return f"v4l2src device={dev}"
    raise ValueError(f"unsupported source type: {source_type!r}")


def build_infer_pipeline(
    source_type: str,
    source: Union[int, str],
    preprocess_cfg: PreprocessCfg,
    infer_cfg: InferCfg,
    postprocess_cfg: PostprocessCfg,
    appsink_name: str,
    display_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Build a full inference GStreamer launch string ending in an appsink.

    ``display_size`` adds an RGA ``dxscale`` on the *display* branch (after
    postprocess); detections stay in original-frame coordinates because
    ``dxpostprocess`` already removes the letterbox.
    """

    q = "queue max-size-buffers=1"
    src = _source_chain(source_type, source)
    pre = _preprocess_element(preprocess_cfg)
    inf = _infer_element(infer_cfg, preprocess_cfg.preprocess_id)
    post = _postprocess_element(postprocess_cfg)

    tail = ""
    if display_size is not None:
        w, h = display_size
        tail = f" ! {q} ! dxscale width={w} height={h}"

    # dxpostprocess only attaches metadata; the buffer pixels are still the
    # decoder's native format (NV12/I420). Convert to RGB with explicit caps so
    # the appsink delivers deterministic 3-channel frames for the Qt display.
    tail += " ! videoconvert ! video/x-raw,format=RGB"

    return (
        f"{src} ! {q} ! {pre} ! {q} ! {inf} ! {q} ! {post}{tail} ! {q} ! "
        f"appsink name={appsink_name} {_APPSINK_OPTS}"
    )
