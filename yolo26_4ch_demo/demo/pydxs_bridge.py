"""Bridge to the native ``pydxs`` GStreamer metadata API.

On the board, ``pydxs`` exposes detection metadata attached to GStreamer buffers
via ``pydxs.dx_get_frame_meta(hash(buffer))``. On the dev host the module is
absent, so the bridge degrades gracefully (every read returns an empty array)
which lets the rest of the pipeline wiring be unit-tested without HW.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .meta_adapter import frame_meta_to_detections

logger = logging.getLogger(__name__)

_EMPTY = np.zeros((0, 6), dtype=np.float32)


def _try_import_pydxs():
    try:
        import pydxs  # type: ignore

        return pydxs
    except Exception as exc:  # pragma: no cover - host path
        logger.warning("pydxs unavailable, native detections disabled: %s", exc)
        return None


class PydxsBridge:
    """Reads detections from a GStreamer buffer via pydxs, with host fallback.

    ``pydxs_module`` is injectable for testing; pass ``None`` to force the
    unavailable path, or omit it to auto-import the real module.
    """

    _SENTINEL = object()

    def __init__(self, pydxs_module=_SENTINEL):
        if pydxs_module is PydxsBridge._SENTINEL:
            pydxs_module = _try_import_pydxs()
        self._pydxs = pydxs_module

    @property
    def available(self) -> bool:
        return self._pydxs is not None

    def detections_for_buffer(self, gst_buffer) -> np.ndarray:
        """Return ``(N, 6)`` detections for a buffer, never raising."""
        if self._pydxs is None:
            return _EMPTY.copy()
        try:
            frame_meta = self._pydxs.dx_get_frame_meta(hash(gst_buffer))
            return frame_meta_to_detections(frame_meta)
        except Exception as exc:  # pragma: no cover - board-only path
            logger.debug("pydxs meta read failed: %s", exc)
            return _EMPTY.copy()
