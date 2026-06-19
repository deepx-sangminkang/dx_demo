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

# Keep only the newest decoded sample, never block, and emit ``new-sample`` so
# the Python side can pull frames + metadata. ``sync`` toggles the appsink's
# clock sync: ``sync=false`` delivers frames as fast as the pipeline produces
# them. NOTE: native-fps pacing is done in Python (StreamPipeline._pace), not via
# ``sync=true`` -- a clock-synced appsink stalls the gapless SEGMENT-loop seek on
# the dx_stream pipeline for several seconds (a visible gap when a clip loops), so
# looping channels must run the appsink ``sync=false`` and pace in Python instead.
_APPSINK_BASE_OPTS = "drop=true max-buffers=1 emit-signals=true"


def _appsink_opts(sync: bool) -> str:
    return f"{_APPSINK_BASE_OPTS} sync={'true' if sync else 'false'}"


# Valid NV12->RGB colour-conversion backends for the display branch.
#   "cpu" : software ``videoconvert`` (works everywhere).
#   "rga" : RK3588 RGA ``dxconvert`` (offloads the conversion to hardware).
_COLOR_CONVERT_ELEMENTS = {"cpu": "videoconvert", "rga": "dxconvert"}


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


# dx_stream GStreamer elements the native backend requires (no SW fallback).
REQUIRED_ELEMENTS = ("dxpreprocess", "dxinfer", "dxpostprocess")


def missing_native_requirements(element_available, pydxs_available: bool):
    """Return a list of human-readable missing dependencies for the dxstream backend.

    ``element_available`` is a ``callable(name) -> bool`` checking the GStreamer
    registry. An empty list means every requirement is satisfied.
    """
    missing = []
    for name in REQUIRED_ELEMENTS:
        if not element_available(name):
            missing.append(f"{name} (dx_stream GStreamer plugin)")
    if not pydxs_available:
        missing.append("pydxs (dx_stream Python bindings)")
    return missing


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
    # A ``video/x-raw`` capsfilter is pinned right after ``decodebin`` so only
    # the decoded *video* pad is linked downstream. Many clips also carry audio
    # (AAC) or data (timecode) tracks; without this filter decodebin's auto
    # linker can grab whichever pad prerolls first, and any non-video pad it
    # exposes is left unlinked. Under the 4-channel VPU contention on RK3588 an
    # unlinked demuxer pad returns GST_FLOW_NOT_LINKED during the gapless loop
    # seek, which qtdemux reports as "Internal data stream error (-5)" and wedges
    # the tile. Filtering to video keeps looping robust (and HW decode intact,
    # since decodebin still selects mppvideodec to produce video/x-raw).
    if source_type == "video":
        return f"filesrc location={source} ! decodebin ! video/x-raw"
    if source_type == "rtsp":
        return (
            f"rtspsrc location={source} latency=100 ! "
            f"rtph264depay ! h264parse ! decodebin ! video/x-raw"
        )
    if source_type == "camera":
        text = str(source)
        dev = text if text.startswith("/dev/") else f"/dev/video{text}"
        return f"v4l2src device={dev}"
    raise ValueError(f"unsupported source type: {source_type!r}")


def meta_source_name(appsink_name: str) -> str:
    """Name of the element (``dxpostprocess``) whose src pad carries DXFrameMeta.

    The detection metadata that ``dxpostprocess`` attaches lives on the
    decoder-native buffer; the downstream colour convert (NV12->RGB) allocates a
    new buffer that does not carry the custom meta, so the meta must be read on
    ``dxpostprocess``'s src pad (before any display ``dxscale`` and the convert)
    and correlated to the appsink frame by buffer PTS. Reading it here also keeps
    detections in the original decoded-frame coordinate space.
    """

    return f"dxmeta_{appsink_name}"


def build_infer_pipeline(
    source_type: str,
    source: Union[int, str],
    preprocess_cfg: PreprocessCfg,
    infer_cfg: InferCfg,
    postprocess_cfg: PostprocessCfg,
    appsink_name: str,
    display_size: Optional[Tuple[int, int]] = None,
    color_convert: str = "cpu",
    sync: bool = False,
    osd: bool = False,
) -> str:
    """Build a full inference GStreamer launch string ending in an appsink.

    ``display_size`` adds an RGA ``dxscale`` on the *display* branch (after the
    detection probe) to downscale only the frame delivered to Qt; detection
    coordinates are unaffected because they are read upstream of ``dxscale`` and
    ``dxpostprocess`` already removes the letterbox (original-frame coords).

    ``color_convert`` selects the NV12->RGB backend: ``"cpu"`` (software
    ``videoconvert``) or ``"rga"`` (RK3588 ``dxconvert``, offloaded to RGA HW).

    ``sync`` controls the appsink clock sync: ``False`` runs at max throughput,
    ``True`` paces output to the source's native frame rate (smooth playback).

    ``osd`` inserts a ``dxosd`` element that renders the segmentation
    masks/boxes onto the frame in HW (matching the dx_stream
    ``run_yolo26n-seg.sh`` reference pipeline). It is placed *after* any display
    ``dxscale`` so the overlay is drawn on the small downscaled frame rather than
    the full decoder-native frame -- dxosd scales every draw coordinate by
    ``frame_meta->_width / buffer_width`` (it reads the live buffer size and the
    DXFrameMeta passes through dxscale unchanged), so masks/boxes still land
    correctly while the per-pixel blend cost drops to the displayed resolution.
    It stays before the NV12->RGB convert (which drops the custom DXFrameMeta
    dxosd needs). The Qt front end then only tiles the already-overlaid RGB
    frames.
    """

    if color_convert not in _COLOR_CONVERT_ELEMENTS:
        raise ValueError(
            f"color_convert must be one of {tuple(_COLOR_CONVERT_ELEMENTS)}, "
            f"got {color_convert!r}"
        )
    convert_element = _COLOR_CONVERT_ELEMENTS[color_convert]

    q = "queue max-size-buffers=1"
    src = _source_chain(source_type, source)
    pre = _preprocess_element(preprocess_cfg)
    inf = _infer_element(infer_cfg, preprocess_cfg.preprocess_id)
    post = _postprocess_element(postprocess_cfg)

    # Detections are always probed on the dxpostprocess src pad so they remain
    # in the *original* decoded-frame coordinate space (and the original frame
    # size can be read from this pad's caps). Any display ``dxscale`` is inserted
    # strictly downstream and only resizes the displayed pixels, never the
    # metadata coordinates.
    meta_name = meta_source_name(appsink_name)
    post = f"{post} name={meta_name}"
    tail = ""
    # Display downscale first (RGA dxscale): shrink the frame to the display
    # resolution before the overlay so dxosd blends masks over far fewer pixels.
    if display_size is not None:
        w, h = display_size
        tail += f" ! {q} ! dxscale width={w} height={h}"
    # Render the segmentation overlay in HW (dxosd) on the downscaled frame, but
    # before the NV12->RGB convert, so the masks are baked into the frame the
    # appsink delivers to Qt. dxosd rescales every draw coordinate to the live
    # buffer size, so it renders correctly even though the DXFrameMeta still
    # carries original-frame coordinates.
    if osd:
        tail += f" ! {q} ! dxosd"

    # dxpostprocess/dxscale only attach metadata or resize; the buffer pixels are
    # still the decoder's native format (NV12/I420). Convert to RGB with explicit
    # caps so the appsink delivers deterministic 3-channel frames for Qt.
    tail += f" ! {convert_element} ! video/x-raw,format=RGB"

    return (
        f"{src} ! {q} ! {pre} ! {q} ! {inf} ! {q} ! {post}{tail} ! {q} ! "
        f"appsink name={appsink_name} {_appsink_opts(sync)}"
    )
