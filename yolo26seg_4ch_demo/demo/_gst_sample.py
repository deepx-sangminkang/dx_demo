"""Extract an RGB numpy frame + GStreamer buffer from an appsink GstSample.

Board-only: it touches the live GStreamer buffer memory. Kept import-clean on
the host (no top-level ``gi``) so the rest of the wiring stays unit-testable.

The appsink is fed by ``dxpostprocess`` (or ``dxscale``); caps carry the frame
geometry and format. We return:
  * ``frame``  : HxWx3 uint8 ndarray (RGB), a copy detached from buffer memory
  * ``buffer`` : the GstBuffer, whose ``hash()`` keys pydxs detection metadata
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np


_CAPS_INT_RE = {
    "width": re.compile(r"width=\(int\)(\d+)"),
    "height": re.compile(r"height=\(int\)(\d+)"),
}
_CAPS_FORMAT_RE = re.compile(r"format=\(string\)([A-Za-z0-9_]+)")


def _structure_int(structure, caps, key):
    """Read an int field, falling back to parsing the caps string.

    ``Structure.get_int`` is a plain GIR method and normally works, but if the
    structure comes from a foreign GI registry (e.g. the dxstream native display
    path) the bound method can raise ``TypeError``. Parsing ``caps.to_string()``
    sidesteps the per-method ``isinstance`` checks entirely.
    """
    try:
        ok, value = structure.get_int(key)
        if ok:
            return value
    except Exception:  # noqa: BLE001 - any GI binding mismatch
        pass
    try:
        match = _CAPS_INT_RE[key].search(caps.to_string() or "")
    except Exception:  # noqa: BLE001
        return None
    return int(match.group(1)) if match else None


def _structure_format(structure, caps):
    """Read the ``format`` field, falling back to parsing the caps string.

    ``Structure.get_string`` is overridden by PyGObject with a strict
    ``isinstance(self, Gst.Structure)`` guard; a structure from a foreign GI
    registry trips it with ``Expected Gst.Structure, but got
    gi.repository.Gst.Structure``. Parse the caps string in that case.
    """
    try:
        fmt = structure.get_string("format")
        if fmt:
            return fmt
    except Exception:  # noqa: BLE001 - any GI binding mismatch
        pass
    try:
        match = _CAPS_FORMAT_RE.search(caps.to_string() or "")
    except Exception:  # noqa: BLE001
        return None
    return match.group(1) if match else None


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

    width = _structure_int(structure, caps, "width")
    height = _structure_int(structure, caps, "height")
    if width is None or height is None:
        return None, buffer

    fmt = _structure_format(structure, caps) or "RGB"
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
