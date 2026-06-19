"""Tests for the render-backend video widget factory (CPU/GPU selection)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from demo import main as demo_main  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_cpu_backend_returns_cpu_widget(qapp):
    w = demo_main.create_video_widget("cpu")
    assert isinstance(w, demo_main.VideoWidget)


def test_auto_backend_returns_some_render_widget(qapp):
    w = demo_main.create_video_widget("auto")
    assert isinstance(w, demo_main._VideoRenderMixin)


def test_gpu_backend_falls_back_to_cpu_when_unavailable(qapp):
    if not demo_main._OPENGL_WIDGET_AVAILABLE:
        w = demo_main.create_video_widget("gpu")
        assert isinstance(w, demo_main.VideoWidget)
    else:
        w = demo_main.create_video_widget("gpu")
        assert isinstance(w, demo_main._VideoRenderMixin)


def test_widget_stores_frame_and_detection_size(qapp):
    import numpy as np

    w = demo_main.create_video_widget("cpu")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    w.set_frame(frame, "rgb")
    w.set_detection_source_size(1920, 1080)
    assert w._latest_frame is frame
    assert w._det_src_size == (1920, 1080)
