"""Import smoke test for the board-only GstSample extractor.

The body runs only with the GStreamer runtime; here we just ensure the module
imports cleanly on the dev host (no top-level ``gi`` import) and exposes the API.
"""

from __future__ import annotations

from demo import _gst_sample as gs


def test_module_exposes_extractor():
    assert hasattr(gs, "extract_frame_and_buffer")
    assert callable(gs.extract_frame_and_buffer)
