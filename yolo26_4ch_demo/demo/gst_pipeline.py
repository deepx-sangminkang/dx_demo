"""GStreamer-based hardware video decoding helpers.

This module builds GStreamer pipeline strings that offload video decoding
onto platform hardware decoders (RK3588 VPU, Intel VAAPI) and is
consumed by ``CaptureThread`` through ``cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)``.

The pipeline terminates with an ``appsink`` emitting either ``video/x-raw,format=BGR``
(default) or ``video/x-raw,format=RGB`` when the RGA-backed ``dxconvert`` element is
used on RK3588. The resulting colour order is reported back through ``open_capture``
so the rest of the demo can skip redundant ``cvtColor`` calls (RGB end-to-end).

All decision logic here is pure/inject-friendly so it can be unit tested
without any real hardware decoder present.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

import cv2

# Appsink shared properties: keep only the newest frame, never block, ignore clock.
_APPSINK = "appsink drop=true max-buffers=1 sync=false"
_BGR_CAPS = "video/x-raw,format=BGR"
_RGB_CAPS = "video/x-raw,format=RGB"

# Captured-frame colour orders surfaced to the rest of the demo.
COLOR_BGR = "bgr"
COLOR_RGB = "rgb"

# RGA-backed colour-convert element shipped by the dx_stream GStreamer plugin.
_RGA_CONVERT_ELEMENT = "dxconvert"

_VALID_SOURCE_TYPES = {"video", "rtsp", "camera"}
_VALID_DECODE_MODES = {"auto", "hw", "sw"}


class Platform(enum.Enum):
    """Detected hardware decode platform."""

    RK3588 = "rk3588"
    INTEL_VAAPI = "intel_vaapi"
    UNKNOWN = "unknown"


class HwDecodeUnavailable(RuntimeError):
    """Raised when a HW decode pipeline cannot be built for the platform."""


@dataclass
class PlatformProbe:
    """Injectable system facts used to determine the HW decode platform."""

    device_tree_compatible: str = ""
    available_elements: Set[str] = field(default_factory=set)
    dri_render_nodes: List[str] = field(default_factory=list)


@dataclass
class DecodeDecision:
    """Result of deciding whether to use HW decoding."""

    use_hw: bool
    platform: Platform
    reason: str


# ===== OpenCV capability detection =====


def opencv_has_gstreamer(build_info: Optional[str] = None) -> bool:
    """Return True if the active OpenCV build was compiled with GStreamer support."""

    if build_info is None:
        build_info = cv2.getBuildInformation()

    for line in build_info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GStreamer:"):
            value = stripped.split(":", 1)[1].strip()
            return value.upper().startswith("YES")
    return False


# ===== Platform detection =====


def _gst_element_available(name: str) -> bool:
    """Check element availability via gi (preferred) or gst-inspect-1.0 fallback."""

    try:
        import gi  # type: ignore

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore

        if not Gst.is_initialized():
            Gst.init(None)
        return Gst.ElementFactory.find(name) is not None
    except Exception:
        pass

    import shutil
    import subprocess

    if shutil.which("gst-inspect-1.0") is None:
        return False
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def gst_element_available(name: str) -> bool:
    """Public wrapper for GStreamer element availability checks."""

    return _gst_element_available(name)


def probe_platform(elements_to_check: Optional[Iterable[str]] = None) -> PlatformProbe:
    """Collect system facts needed for platform detection from the live system."""

    compatible = ""
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compatible = f.read().decode("utf-8", errors="ignore")
    except OSError:
        compatible = ""

    if elements_to_check is None:
        elements_to_check = (
            "mppvideodec",
            "vaapidecodebin",
            "vah264dec",
            _RGA_CONVERT_ELEMENT,
        )
    available = {name for name in elements_to_check if _gst_element_available(name)}

    render_nodes: List[str] = []
    dri_dir = "/dev/dri"
    try:
        render_nodes = [
            os.path.join(dri_dir, n)
            for n in os.listdir(dri_dir)
            if n.startswith("renderD")
        ]
    except OSError:
        render_nodes = []

    return PlatformProbe(
        device_tree_compatible=compatible,
        available_elements=available,
        dri_render_nodes=render_nodes,
    )


def detect_platform(probe: Optional[PlatformProbe] = None) -> Platform:
    """Determine the HW decode platform from a (possibly injected) probe."""

    if probe is None:
        probe = probe_platform()

    if "rk3588" in probe.device_tree_compatible:
        return Platform.RK3588

    has_vaapi_element = bool(
        {"vaapidecodebin", "vah264dec"} & probe.available_elements
    )
    if has_vaapi_element and probe.dri_render_nodes:
        return Platform.INTEL_VAAPI

    return Platform.UNKNOWN


def rga_convert_available(platform: Optional[Platform] = None) -> bool:
    """Return True when the RGA-backed ``dxconvert`` element can offload colour
    conversion on RK3588.

    Only RK3588 ships the RGA hardware that ``dxconvert`` accelerates; on other
    platforms the element (even if present) would fall back to libyuv, so we
    keep using the standard ``videoconvert`` chain there.
    """

    if platform is None:
        platform = detect_platform()
    if platform != Platform.RK3588:
        return False
    return _gst_element_available(_RGA_CONVERT_ELEMENT)


# ===== Decode decision (fallback policy) =====


def resolve_decode(
    decode_mode: str,
    opencv_gstreamer: bool,
    platform: Platform,
) -> DecodeDecision:
    """Decide whether HW decoding should be used, with graceful fallback.

    - ``sw``  : always software decode.
    - ``hw``  : hardware decode required; falls back to SW (with reason) when
                prerequisites are missing instead of crashing.
    - ``auto``: hardware decode when possible, otherwise software.
    """

    mode = (decode_mode or "auto").lower()
    if mode not in _VALID_DECODE_MODES:
        mode = "auto"

    if mode == "sw":
        return DecodeDecision(False, platform, "decode mode set to 'sw'")

    if not opencv_gstreamer:
        return DecodeDecision(
            False, platform, "OpenCV built without GStreamer support; using SW decode"
        )

    if platform == Platform.UNKNOWN:
        return DecodeDecision(
            False, platform, "no supported HW decoder detected; using SW decode"
        )

    return DecodeDecision(True, platform, f"using HW decode on {platform.value}")


# ===== Pipeline construction =====


def _decoder_chain(
    platform: Platform,
    rga_convert: bool = False,
    scale_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Decoder + color-convert element chain feeding the appsink tail.

    When ``rga_convert`` is set on RK3588, the RGA-backed ``dxconvert`` element
    performs the NV12->RGB conversion on hardware (offloading the CPU) and the
    caller emits ``RGB`` caps; otherwise the standard CPU ``videoconvert`` is
    used and ``BGR`` caps are emitted.

    When ``scale_size`` is provided together with ``rga_convert`` on RK3588, the
    RGA-backed ``dxscale`` element resizes the (NV12) frame to the model input
    size on hardware *before* the colour conversion, removing the CPU
    ``cv2.resize`` from the preprocessing hot path. ``dxscale`` is scale-only
    (no aspect-ratio padding), so callers must treat the result as a stretched
    resize when remapping detection coordinates.
    """

    if platform == Platform.RK3588:
        # mppvideodec outputs NV12.
        if rga_convert:
            # Optional RGA HW resize (NV12) before the NV12->RGB conversion.
            if scale_size is not None:
                w, h = scale_size
                return f"mppvideodec ! dxscale width={w} height={h} ! dxconvert"
            # dxconvert (librga) does NV12->RGB on the RGA hardware.
            return "mppvideodec ! dxconvert"
        # mpp's RGA-backed videoconvert handles NV12->BGR.
        return "mppvideodec ! videoconvert"
    if platform == Platform.INTEL_VAAPI:
        # vaapidecodebin handles parse+decode; vapostproc keeps conversion on GPU.
        return "vaapidecodebin ! vapostproc ! videoconvert"
    raise HwDecodeUnavailable(f"no HW decoder chain for platform {platform}")


def hw_output_color_format(
    source_type: str, platform: Platform, rga_convert: bool = False
) -> str:
    """Colour order of frames produced by the HW pipeline.

    Only the RK3588 ``dxconvert`` path on file/RTSP sources emits ``RGB``; every
    other HW path emits ``BGR``.
    """

    if (
        rga_convert
        and platform == Platform.RK3588
        and source_type in {"video", "rtsp"}
    ):
        return COLOR_RGB
    return COLOR_BGR


def _camera_device(source: Union[int, str]) -> str:
    """Normalise a camera source (index or path) into a /dev/video* path."""

    if isinstance(source, int):
        return f"/dev/video{source}"
    text = str(source)
    if text.isdigit():
        return f"/dev/video{text}"
    return text


def build_gst_pipeline(
    source_type: str,
    source: Union[int, str],
    platform: Platform,
    rga_convert: bool = False,
    scale_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Build a GStreamer launch string for HW-accelerated decoding.

    The appsink emits ``RGB`` when the RGA ``dxconvert`` path is selected on
    RK3588 (file/RTSP), otherwise ``BGR``.

    When ``scale_size=(w, h)`` is given on the RK3588 RGA (file/RTSP) path, a
    ``dxscale`` element resizes frames to ``w x h`` on RGA hardware and the
    appsink caps are pinned to that resolution, offloading the CPU
    ``cv2.resize`` from preprocessing. ``scale_size`` is ignored on paths that
    do not use ``dxconvert`` (camera, non-RK3588, ``rga_convert=False``).
    """

    if platform == Platform.UNKNOWN:
        raise HwDecodeUnavailable("cannot build HW decode pipeline on unknown platform")

    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(f"unsupported source type: {source_type!r}")

    color_format = hw_output_color_format(source_type, platform, rga_convert)
    # dxscale only applies on the RGA dxconvert (RGB) path; pin caps to the
    # scaled resolution there so negotiation with dxscale is explicit.
    use_scale = scale_size is not None and color_format == COLOR_RGB
    caps = _RGB_CAPS if color_format == COLOR_RGB else _BGR_CAPS
    if use_scale:
        w, h = scale_size
        caps = f"{caps},width={w},height={h}"
    tail = f"{caps} ! {_APPSINK}"

    chain_scale = scale_size if use_scale else None

    if source_type == "video":
        decoder = _decoder_chain(platform, rga_convert, chain_scale)
        return (
            f"filesrc location={source} ! parsebin ! {decoder} ! {tail}"
        )

    if source_type == "rtsp":
        decoder = _decoder_chain(platform, rga_convert, chain_scale)
        # depay/parse are codec specific; default to H.264. H.265 streams can be
        # supported later by branching on caps.
        return (
            f"rtspsrc location={source} latency=100 ! "
            f"rtph264depay ! h264parse ! {decoder} ! {tail}"
        )

    # camera: keep the simple CPU videoconvert path (v4l2 raw formats are not
    # guaranteed to be dxconvert-compatible), always producing BGR.
    device = _camera_device(source)
    return f"v4l2src device={device} ! videoconvert ! {tail}"


# ===== Capture opening with graceful fallback =====


def open_capture(
    source: Union[int, str],
    source_type: str,
    decode_mode: str,
    platform: Platform,
    opencv_gstreamer: bool,
    rga_convert: bool = False,
    scale_size: Optional[Tuple[int, int]] = None,
    video_capture_factory: Optional[Callable[..., "cv2.VideoCapture"]] = None,
) -> Tuple["cv2.VideoCapture", bool, str, str]:
    """Open a ``cv2.VideoCapture``, preferring HW decode with SW fallback.

    Returns ``(capture, used_hw, reason, color_format)``. ``color_format`` is
    ``"rgb"`` only when the RGA ``dxconvert`` path is actually used, otherwise
    ``"bgr"``. ``capture`` may be unopened if even the software fallback fails;
    the caller is expected to check ``isOpened()``.

    ``scale_size`` (when set) requests RGA HW resize via ``dxscale`` on the
    RK3588 RGB path; it is ignored on every other decode path.
    """

    if video_capture_factory is None:
        video_capture_factory = cv2.VideoCapture

    decision = resolve_decode(decode_mode, opencv_gstreamer, platform)

    if decision.use_hw:
        try:
            pipeline = build_gst_pipeline(
                source_type, source, platform, rga_convert, scale_size
            )
            cap = video_capture_factory(pipeline, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                color_format = hw_output_color_format(
                    source_type, platform, rga_convert
                )
                return cap, True, decision.reason, color_format
            reason = (
                f"HW decode pipeline failed to open; falling back to SW "
                f"(platform={platform.value})"
            )
        except (HwDecodeUnavailable, ValueError) as exc:
            reason = f"HW decode unavailable ({exc}); falling back to SW"
    else:
        reason = decision.reason

    cap = video_capture_factory(source)
    return cap, False, reason, COLOR_BGR
