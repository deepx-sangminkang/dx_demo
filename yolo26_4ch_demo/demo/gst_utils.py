"""Small GStreamer helpers shared by the dx_stream backend.

Intentionally free of OpenCV and Qt imports so it can be unit-tested and reused
from the pure-pipeline modules.
"""

from __future__ import annotations


def gst_element_available(name: str) -> bool:
    """Return True if a GStreamer element factory ``name`` is registered.

    Prefers the in-process ``gi`` registry and falls back to the
    ``gst-inspect-1.0`` CLI when the Python bindings are unavailable.
    """

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
