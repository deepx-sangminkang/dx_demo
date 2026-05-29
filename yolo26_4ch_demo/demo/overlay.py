"""Pure geometry helpers for overlaying detections on scaled video frames.

Kept free of Qt/OpenCV imports so the coordinate mapping can be unit tested
without a GUI environment.
"""

from __future__ import annotations

from typing import Sequence, Tuple


def scale_box(
    box: Sequence[float],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    offset: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Map a box from source-image coords to scaled-pixmap widget coords.

    Args:
        box: ``(x1, y1, x2, y2, ...)`` in source-image pixel coordinates.
        src_size: ``(width, height)`` of the source image.
        dst_size: ``(width, height)`` of the scaled pixmap shown on screen.
        offset: ``(x, y)`` top-left position of the pixmap within the widget.

    Returns:
        ``(x1, y1, x2, y2)`` in widget coordinates.
    """

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    off_x, off_y = offset

    sx = dst_w / src_w if src_w else 1.0
    sy = dst_h / src_h if src_h else 1.0

    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return (
        off_x + x1 * sx,
        off_y + y1 * sy,
        off_x + x2 * sx,
        off_y + y2 * sy,
    )
