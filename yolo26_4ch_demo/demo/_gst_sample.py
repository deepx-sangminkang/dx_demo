"""Extract an RGB numpy frame + GStreamer buffer from an appsink GstSample.

Board-only: it touches the live GStreamer buffer memory. Kept import-clean on
the host (no top-level ``gi``) so the rest of the wiring stays unit-testable.

The appsink is fed by ``dxpostprocess`` (or ``dxscale``); caps carry the frame
geometry and format. We return:
  * ``frame``  : HxWx3 uint8 ndarray (RGB), a copy detached from buffer memory
  * ``buffer`` : the GstBuffer, whose ``hash()`` keys pydxs detection metadata
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def extract_frame_and_buffer(
    sample,
) -> Tuple[Optional[np.ndarray], Optional[object]]:  # pragma: no cover - board only
    """Return ``(rgb_frame, gst_buffer)`` from a GstSample, or ``(None, None)``."""
    if sample is None:
        return None, None

    from gi.repository import Gst  # type: ignore  # noqa: F401

    buffer = sample.get_buffer()
    caps = sample.get_caps()
    if buffer is None or caps is None:
        return None, None

    # Guard against empty/unnegotiated caps so we never call
    # gst_caps_get_structure / gst_structure_get_int on NULL (which would emit
    # the "GST_IS_CAPS failed" / "structure != NULL" GLib criticals).
    if caps.get_size() < 1:
        return None, buffer
    structure = caps.get_structure(0)
    if structure is None:
        return None, buffer

    ok_w, width = structure.get_int("width")
    ok_h, height = structure.get_int("height")
    if not (ok_w and ok_h):
        return None, buffer

    fmt = structure.get_string("format") or "RGB"
    if fmt not in ("RGB", "BGR"):
        # The pipeline pins format=RGB before the appsink; anything else means
        # the convert element was dropped. Skip rather than misinterpret bytes.
        return None, buffer

    ok, mapinfo = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return None, buffer
    try:
        raw = np.frombuffer(mapinfo.data, dtype=np.uint8)
        channels = 3
        expected = width * height * channels
        if raw.size < expected:
            return None, buffer
        frame = raw[:expected].reshape(height, width, channels)
        if fmt == "BGR":
            frame = frame[:, :, ::-1]
        frame = np.ascontiguousarray(frame)  # detach from mapped memory
    finally:
        buffer.unmap(mapinfo)

    return frame, buffer
