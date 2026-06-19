"""Unit tests for demo.overlay.scale_box coordinate mapping."""

from __future__ import annotations

from demo.overlay import scale_box


def test_scale_box_identity_no_scaling_no_offset():
    box = (10, 20, 30, 40)
    out = scale_box(box, (100, 100), (100, 100), (0, 0))
    assert out == (10.0, 20.0, 30.0, 40.0)


def test_scale_box_half_scale_with_offset():
    # Source 200x100 displayed at 100x50, placed at offset (5, 7).
    box = (20, 10, 60, 30)
    x1, y1, x2, y2 = scale_box(box, (200, 100), (100, 50), (5, 7))
    assert (x1, y1, x2, y2) == (5 + 10.0, 7 + 5.0, 5 + 30.0, 7 + 15.0)


def test_scale_box_handles_zero_source_dimension():
    # Degenerate source size must not raise; falls back to scale 1.0.
    out = scale_box((1, 2, 3, 4), (0, 0), (50, 50), (0, 0))
    assert out == (1.0, 2.0, 3.0, 4.0)


def test_scale_box_ignores_extra_box_fields():
    # Detections carry (x1,y1,x2,y2,score,class); extra fields are ignored.
    box = (10, 10, 20, 20, 0.9, 3)
    out = scale_box(box, (100, 100), (200, 200), (0, 0))
    assert out == (20.0, 20.0, 40.0, 40.0)
