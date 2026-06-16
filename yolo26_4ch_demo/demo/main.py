"""YOLO26 multi-channel Qt demo (dx_stream backend).

- One native dx_stream GStreamer pipeline per channel
  (decodebin -> dxpreprocess -> dxinfer -> dxpostprocess -> appsink),
  detections read back via pydxs, displayed in a Qt 2x2 grid.
- Inference runs entirely on the NPU inside GStreamer; the Python side only
  decodes display frames and draws overlays. No OpenCV dependency.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to sys.path (so it can be run from anywhere)
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import yaml
from PySide6 import QtCore, QtGui, QtWidgets

# QOpenGLWidget lives in a separate module and may be unavailable if the Qt
# build lacks OpenGL support; the GPU render path degrades to CPU QPainter then.
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget  # type: ignore

    _OPENGL_WIDGET_AVAILABLE = True
except Exception:  # pragma: no cover - depends on Qt build
    QOpenGLWidget = None  # type: ignore
    _OPENGL_WIDGET_AVAILABLE = False

from demo.engine import NativeDisplayMeta
from demo.overlay import scale_box
from demo import native_config, native_pipeline, native_signal
from demo import cpu_affinity
from demo.gst_utils import gst_element_available
from demo.meta_adapter import filter_by_classes
from demo.pydxs_bridge import PydxsBridge
from demo.stream_pipeline import StreamPipeline


# ===== Video render widget (CPU QPainter + optional Mali GPU/OpenGL) =====


class _VideoRenderMixin:
    """Shared drawing logic for the per-channel video widgets.

    Holds the latest frame, detections and stats, and renders them with a
    QPainter. Two concrete widgets reuse this: ``VideoWidget`` (CPU raster) and
    ``GLVideoWidget`` (QOpenGLWidget, Mali GPU accelerated). Both call
    ``_render`` from their paint entry point so the visuals are identical.

    - set_frame(np.ndarray) only stores the latest frame; actual drawing is done
      in the paint callback. This decouples external update frequency from the
      timer-driven repaint, giving smooth playback.
    """

    def _init_render_state(self) -> None:
        self.setMinimumSize(320, 240)
        self._latest_frame: Optional[np.ndarray] = None
        # Colour order of the stored frame ("bgr" or "rgb").
        self._frame_color_format: str = "bgr"
        # Latest detections (Nx6: x1,y1,x2,y2,score,class) in original-image
        # coordinates, overlaid in paintEvent. Updated asynchronously from the
        # inference pipeline, decoupled from the displayed frame.
        self._detections: Optional[np.ndarray] = None
        self._overlay_classes: List[str] = []
        self._overlay_palette: Optional[np.ndarray] = None
        # Original resolution of the detection coordinate space (the decoded
        # frame size, before any display-branch downscale). When the displayed
        # frame is downscaled, boxes must be mapped from this size, not from the
        # smaller displayed-frame size. None -> fall back to the displayed size.
        self._det_src_size: Optional[tuple] = None
        # Text overlay for per-channel statistics
        # (FPS, processed frame count, dropped frame count)
        self._stats_text: str = ""

    def set_overlay_style(self, class_names: List[str], palette: np.ndarray) -> None:
        """Inject class names and the colour palette used to draw boxes/labels."""

        self._overlay_classes = class_names
        self._overlay_palette = palette

    def set_detections(self, detections: Optional[np.ndarray]) -> None:
        """Store the latest detections (original-image coords) for overlay."""

        self._detections = detections

    def set_detection_source_size(self, width: int, height: int) -> None:
        """Set the original resolution the detection coordinates are expressed in.

        Used to map boxes correctly when the displayed frame is downscaled.
        """

        if width > 0 and height > 0:
            self._det_src_size = (int(width), int(height))

    @QtCore.Slot(np.ndarray)
    def set_frame(self, frame: np.ndarray, color_format: str = "bgr") -> None:
        """Receives an image and stores it in the internal buffer.

        ``color_format`` records whether the frame is BGR (default) or RGB so
        ``paintEvent`` can avoid a redundant colour conversion.
        """

        if frame is None:
            return

        self._latest_frame = frame
        self._frame_color_format = color_format

    def update_stats_text(self, text: str) -> None:
        """Stores a stats string passed from outside for overlay use.

        - Drawn on top of the video frame in paintEvent.
        """

        self._stats_text = text

    def _draw_detections_overlay(
        self,
        painter: QtGui.QPainter,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
        off_x: int,
        off_y: int,
    ) -> None:  # pragma: no cover - GUI only
        """Draw detection boxes + labels mapped onto the scaled pixmap."""

        detections = self._detections
        if detections is None or len(detections) == 0:
            return

        painter.setFont(QtGui.QFont("Monospace", 9))
        metrics = QtGui.QFontMetrics(painter.font())

        for det in detections:
            x1, y1, x2, y2 = scale_box(
                det, (src_w, src_h), (dst_w, dst_h), (off_x, off_y)
            )
            class_id = int(det[5])
            score = float(det[4])

            if self._overlay_palette is not None and 0 <= class_id < len(
                self._overlay_palette
            ):
                r, g, b = (int(c) for c in self._overlay_palette[class_id][:3])
            else:
                r, g, b = 0, 255, 0
            color = QtGui.QColor(r, g, b)

            painter.setPen(QtGui.QPen(color, 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(
                int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            )

            if 0 <= class_id < len(self._overlay_classes):
                name = self._overlay_classes[class_id]
            else:
                name = str(class_id)
            label = f"{name}: {score:.2f}"

            text_rect = metrics.boundingRect(label)
            lh = text_rect.height()
            lw = text_rect.width()
            label_y = int(y1) - lh if int(y1) - lh > 0 else int(y1) + lh
            painter.fillRect(int(x1), label_y - lh, lw + 4, lh + 4, color)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            painter.drawText(int(x1) + 2, label_y, label)

    def _render(self, painter: QtGui.QPainter) -> None:  # pragma: no cover - GUI only
        """Paint the latest frame, detection overlay and stats text.

        Shared by both the CPU and GPU widgets. Clears to black first so the
        letterbox margins are correct even on the OpenGL path (which has no
        stylesheet background fill).
        """

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(0, 0, 0))

        if self._latest_frame is not None:
            # dx_stream appsink always delivers RGB frames (RGA dxconvert /
            # videoconvert pins format=RGB), so no colour conversion is needed.
            frame_rgb = self._latest_frame
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            pix = QtGui.QPixmap.fromImage(qimg)

            # Draw scaled to widget size while preserving aspect ratio.
            # FastTransformation avoids the expensive smooth resampling so the
            # paint timer keeps up at high refresh rates (smoother playback). On
            # the GPU widget the scale/compositing runs on the Mali GPU.
            target = pix.scaled(
                rect.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
            x = rect.x() + (rect.width() - target.width()) // 2
            y = rect.y() + (rect.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)

            # Overlay detection boxes/labels mapped from the original detection
            # coordinate space onto the scaled pixmap. When the displayed frame
            # was downscaled, boxes are in the original (larger) resolution, so
            # use that for the mapping; otherwise the displayed frame size is the
            # detection space.
            src_w, src_h = self._det_src_size if self._det_src_size else (w, h)
            self._draw_detections_overlay(
                painter, src_w, src_h, target.width(), target.height(), x, y
            )

        # Stats text overlay on top of the video
        if self._stats_text:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
            painter.setFont(QtGui.QFont("Monospace", 10))
            metrics = QtGui.QFontMetrics(painter.font())
            text_rect = metrics.boundingRect(self._stats_text)
            margin = 4
            bg_rect = QtCore.QRect(
                rect.left() + margin,
                rect.top() + margin,
                text_rect.width() + margin * 2,
                text_rect.height() + margin * 2,
            )
            painter.fillRect(bg_rect, QtGui.QColor(0, 0, 0, 160))
            painter.drawText(
                bg_rect.adjusted(margin, margin, -margin, -margin),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                self._stats_text,
            )


class VideoWidget(_VideoRenderMixin, QtWidgets.QWidget):
    """CPU raster video widget (QPainter on a regular QWidget)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self._init_render_state()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - GUI only
        painter = QtGui.QPainter(self)
        self._render(painter)


if _OPENGL_WIDGET_AVAILABLE:

    class GLVideoWidget(_VideoRenderMixin, QOpenGLWidget):  # type: ignore[misc]
        """Mali GPU accelerated video widget (QPainter over an OpenGL surface).

        Scaling and compositing of the (already RGA-downscaled) tiles run on the
        embedded GPU instead of the CPU raster engine.
        """

        def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
            super().__init__(parent)
            self._init_render_state()

        def paintGL(self) -> None:  # pragma: no cover - GUI only
            painter = QtGui.QPainter(self)
            self._render(painter)

else:  # pragma: no cover - depends on Qt build
    GLVideoWidget = None  # type: ignore


def create_video_widget(render_backend: str = "auto") -> "_VideoRenderMixin":
    """Create a per-channel video widget for the requested render backend.

    ``render_backend`` is one of ``auto`` (GPU if available, else CPU), ``gpu``
    (force the OpenGL widget; falls back to CPU with a warning if unavailable)
    or ``cpu`` (force the QPainter raster widget).
    """

    backend = (render_backend or "auto").strip().lower()
    want_gpu = backend in ("auto", "gpu")
    if want_gpu and _OPENGL_WIDGET_AVAILABLE and GLVideoWidget is not None:
        try:
            return GLVideoWidget()
        except Exception as exc:  # pragma: no cover - GUI only
            print(f"[render] GPU widget init failed ({exc}); using CPU renderer")
    elif backend == "gpu":
        print("[render] OpenGL widget unavailable; falling back to CPU renderer")
    return VideoWidget()


# ===== Main Window =====


class MainWindow(QtWidgets.QMainWindow):
    frame_ready = QtCore.Signal(int, object, dict)  # channel_id, frame, meta (display path)
    detections_ready = QtCore.Signal(int, object, dict)  # channel_id, detections, meta
    source_size_ready = QtCore.Signal(int, int, int)  # channel_id, width, height

    def __init__(self, config: Dict[str, Any], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("YOLO26 Multi-Channel Demo")
        self.config = config
        # Render backend for the video tiles: auto | gpu | cpu.
        self._render_backend = str(config.get("render_backend", "auto"))
        # Set of selected class IDs (written only from GUI thread; workers read-only)
        self._selected_classes: Set[int] = set()
        self._selected_lock = threading.Lock()

        # Accumulated metrics structure for performance measurement
        self._metrics: Dict[str, Any] = {
            "frames": 0,
            "sum_read": 0.0,
            "sum_pre": 0.0,
            "sum_inf": 0.0,
            "sum_draw": 0.0,
            "start_ts": None,
        }
        self._metrics_lock = threading.Lock()

    # Per-channel stats structure (processed frames / FPS)
        self._channel_stats: Dict[int, Dict[str, Any]] = {}
        self._channel_stats_lock = threading.Lock()

        # Buffer to hold the most recent frame per channel
        # - Even if frame_ready signals arrive in bursts from workers,
        #   the GUI always uses only the latest frame per channel.
        self._latest_frames: Dict[int, np.ndarray] = {}
        self._latest_meta: Dict[int, Dict[str, Any]] = {}
        self._latest_lock = threading.Lock()

        # Timer for periodic screen refresh (approx. 30 FPS)
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.timeout.connect(self._on_paint_timer)
        self._paint_timer.start(33)  # 33ms interval ≒ 30 FPS

        # Build central 2x2 grid layout
        central = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(central)
        self.video_widgets: List["_VideoRenderMixin"] = []
        for i in range(4):
            w = create_video_widget(self._render_backend)
            self.video_widgets.append(w)
            row, col = divmod(i, 2)
            grid.addWidget(w, row, col)
        self.setCentralWidget(central)

        # Right-side class filter panel (empty by default; populated after engine is created)
        self.class_dock = QtWidgets.QDockWidget("", self)
        self.class_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        # The title-bar close(X)/float buttons are not used;
        # the panel is controlled only via the side toggle button on the left.
        self.class_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        # Default width range when panel is expanded (clamped to prevent layout breakage).
        self._dock_min_width_expanded = 220
        self._dock_max_width_expanded = 320
        self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
        self.class_dock.setMaximumWidth(self._dock_max_width_expanded)

        # Container: toggle button on the left + actual class panel on the right
        self.dock_outer_widget = QtWidgets.QWidget(self.class_dock)
        outer_layout = QtWidgets.QHBoxLayout(self.dock_outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        # Round toggle button positioned at the left-centre (always visible outside the dock)
        self.side_toggle_button = QtWidgets.QToolButton(self.dock_outer_widget)
        self.side_toggle_button.setFixedSize(28, 28)
        self.side_toggle_button.setCheckable(True)
        self.side_toggle_button.setChecked(True)
        self.side_toggle_button.setAutoRaise(True)
        self.side_toggle_button.setStyleSheet(
            "QToolButton { border-radius: 14px; background-color: #444; color: white; padding: 0px; min-width: 0px; min-height: 0px; max-width: 28px; max-height: 28px; }"
        )
        self.side_toggle_button.setText(">")
        self.side_toggle_button.clicked.connect(self._on_side_toggle_clicked)

        # Use fixed width + fixed size policy so the toggle button takes up minimal panel space
        self.side_toggle_button.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )

        outer_layout.addWidget(self.side_toggle_button, 0, QtCore.Qt.AlignVCenter)

        # Actual class panel content widget (only this part is hidden on toggle)
        self.class_panel_widget = QtWidgets.QWidget(self.dock_outer_widget)
        self.class_panel_layout = QtWidgets.QVBoxLayout(self.class_panel_widget)
        self.class_panel_layout.setContentsMargins(4, 4, 4, 4)
        self.class_panel_layout.setSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear_all)
        btn_row.addStretch(1)

        self.class_panel_layout.addLayout(btn_row)

        # Area that holds the actual checkbox list (container for scroll area)
        self.class_list_container = QtWidgets.QWidget(self.class_panel_widget)
        self.class_list_layout = QtWidgets.QVBoxLayout(self.class_list_container)
        self.class_list_layout.setContentsMargins(0, 0, 0, 0)
        self.class_list_layout.setSpacing(2)
        self.class_list_layout.addStretch(1)

        scroll_area = QtWidgets.QScrollArea(self.class_panel_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.class_list_container)

        scroll_container = QtWidgets.QWidget(self.class_panel_widget)
        scroll_layout = QtWidgets.QHBoxLayout(scroll_container)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)

        scroll_layout.addWidget(scroll_area, 1)

        self.class_panel_layout.addWidget(scroll_container)

        outer_layout.addWidget(self.class_panel_widget, 1)

        self.class_dock.setWidget(self.dock_outer_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.class_dock)

        # Connect frame_ready signal to the VideoWidget update slot
        self.frame_ready.connect(self.on_frame_ready)
        self.detections_ready.connect(self.on_detections_ready)
        self.source_size_ready.connect(self.on_source_size_ready)

        # Sync side button state when the dock widget visibility changes
        self.class_dock.visibilityChanged.connect(self._on_class_dock_visibility_changed)

        # Initialize engine and workers/queues
        self._init_engine_and_workers(config)

    # ----- Class filter -----

    def _build_class_filter_panel(self, class_names: List[str]) -> None:
        """Build checkboxes from a list of class names.

        - For COCO80 there are ~80 entries; a scrollable layout is recommended.
        """

        # Clear existing checkboxes (remove widgets but keep the stretch item)
        for i in reversed(range(self.class_list_layout.count())):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setParent(None)

        # Populate checkboxes from the top
        for class_id, name in enumerate(class_names):
            cb = QtWidgets.QCheckBox(f"{class_id}: {name}")
            cb.setChecked(True)  # all shown by default
            cb.stateChanged.connect(self._on_any_class_checkbox_changed)

            self.class_list_layout.insertWidget(
                self.class_list_layout.count() - 1, cb
            )

        # Build the selected-class set once based on the initial state (all checked)
        self._rebuild_selected_classes()

    def _on_select_all(self) -> None:
        """Set all classes to checked state."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(True)

    def _on_clear_all(self) -> None:
        """Set all classes to unchecked state."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(False)

    def _on_any_class_checkbox_changed(self, state: int) -> None:
        """Re-scan all checkboxes and rebuild the set whenever any class checkbox changes."""

        self._rebuild_selected_classes()

    def _rebuild_selected_classes(self) -> None:
        """Rebuild _selected_classes based on the current state of all checkboxes."""

        selected: Set[int] = set()
        class_id = 0
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                if widget.isChecked():
                    selected.add(class_id)
                class_id += 1

        with self._selected_lock:
            self._selected_classes = selected

    # ----- Performance measurement / logging -----

    def _update_performance_metrics(self, meta: Dict[str, Any]) -> None:
        """Accumulate performance metrics from per-frame meta and periodically print a summary log."""

        t_read = meta.get("t_read", 0.0)
        t_pre = meta.get("t_preprocess", 0.0)
        t_inf = meta.get("t_inference", 0.0)
        t_pp_draw = meta.get("t_draw", 0.0)

        now = time.perf_counter()

        with self._metrics_lock:
            m = self._metrics

            if m["start_ts"] is None:
                m["start_ts"] = now

            m["frames"] += 1
            m["sum_read"] += t_read
            m["sum_pre"] += t_pre
            m["sum_inf"] += t_inf
            m["sum_draw"] += t_pp_draw

            # The old per-frame average log is replaced by the throughput log,
            # so only accumulated stats are maintained here without separate output.

    def _log_throughput_summary(self) -> None:
        """Log the native dx_stream display throughput every 10 seconds."""

        now = time.time()
        # If no last-log timestamp yet, just initialise and return
        last = getattr(self, "_last_throughput_log_ts", None)
        if last is None:
            self._last_throughput_log_ts = now
            return

        if now - last < 10.0:
            return

        self._last_throughput_log_ts = now

        # Inference runs entirely inside the native pipeline; report the
        # aggregate display FPS delivered to the Qt front end.
        count = getattr(self, "_native_frame_count", 0)
        t0 = getattr(self, "_native_fps_t0", None)
        native_fps = count / (now - t0) if (t0 and now > t0) else 0.0
        print(
            "[THROUGHPUT] dxstream native display={:.1f} fps "
            "(total frames={})".format(native_fps, count)
        )

    # ----- Per-channel stats update -----

    def _update_channel_stats_on_frame(self, channel_id: int, meta: Dict[str, Any]) -> None:
        """Update per-channel stats based on frames that reached on_frame_ready.

        - processed_frames: number of frames that actually reached the screen
        - fps: calculated from the increase in processed_frames over the last 1 second
        """

        now = time.perf_counter()

        with self._channel_stats_lock:
            s = self._channel_stats.setdefault(
                channel_id,
                {
                    "processed_frames": 0,
                    "last_ts": now,
                    "last_frames_for_fps": 0,
                    "fps": 0.0,
                },
            )

            s["processed_frames"] += 1

            # FPS calculation: update using elapsed time and frame count since last checkpoint
            elapsed = now - s["last_ts"]
            if elapsed >= 1.0:
                delta_frames = s["processed_frames"] - s["last_frames_for_fps"]
                s["fps"] = delta_frames / elapsed if elapsed > 0 else 0.0
                s["last_ts"] = now
                s["last_frames_for_fps"] = s["processed_frames"]

    def _get_channel_stats_snapshot(self, channel_id: int) -> str:
        """Return current per-channel stats as a compact string.

        e.g. "ch1 fps=29.8 proc=1234"
        """

        with self._channel_stats_lock:
            s = self._channel_stats.get(channel_id)
            if not s:
                fps = 0.0
                proc = 0
            else:
                fps = s["fps"]
                proc = s["processed_frames"]

        return (
            f"ch{channel_id+1} "
            f"fps={fps:.1f} "
            f"proc={proc}"
        )

    # ----- Class panel toggle -----

    def _on_side_toggle_clicked(self, checked: bool) -> None:
        """Show/hide only the class panel content when the left round button is clicked.

        The dock itself stays visible at all times; only the class panel part is hidden
        so the toggle button never disappears.
        """

        self.class_panel_widget.setVisible(checked)

        # When collapsing, shrink dock width to roughly the toggle button width;
        # when expanding, restore the original width range.
        if checked:
            # Expanded: restore the previously configured width range.
            self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
            self.class_dock.setMaximumWidth(self._dock_max_width_expanded)
        else:
            # Collapsed: tightly constrain width so only the toggle button is visible.
            collapsed_width = self.side_toggle_button.width() + 8
            self.class_dock.setMinimumWidth(collapsed_width)
            self.class_dock.setMaximumWidth(collapsed_width)

        # Expanded (checked=True): show '>' (collapses to the right)
        # Hidden: show '<' (expands to the left)
        self.side_toggle_button.setText(">" if checked else "<")

    def _on_class_dock_visibility_changed(self, visible: bool) -> None:
        """Keep button state consistent even when the entire dock is hidden.

        When the dock is fully closed (user pressed X), it needs to be re-opened,
        so here we only tidy up the text without touching side_toggle_button's checked state.
        """

        # Even when the dock is hidden, the button reappears when the dock reopens,
        # so no special extra sync is needed; just align the text for consistency.
        if visible and self.class_panel_widget.isVisible():
            self.side_toggle_button.setText(">")
        elif visible:
            self.side_toggle_button.setText("<")

    def get_selected_classes(self) -> Set[int]:
        """Return the selected class set for workers to read."""

        with self._selected_lock:
            return set(self._selected_classes)

    # ----- Engine and worker initialisation -----

    def _init_engine_and_workers(self, config: Dict[str, Any]) -> None:
        """Initialise the native dx_stream inference pipelines (one per channel)."""

        # Default containers so shutdown is always safe.
        self.stream_pipelines: List[StreamPipeline] = []

        self.engine_backend = native_config.get_engine_backend(config)
        self._init_dxstream_backend(config)

    # ----- Worker → GUI forwarding wrappers -----

    def _init_dxstream_backend(self, config: Dict[str, Any]) -> None:
        """Initialise the native dx_stream inference pipelines (one per channel).

        Inference runs entirely in GStreamer (dxpreprocess -> dxinfer ->
        dxpostprocess); detections are read via pydxs and fed into the Qt
        display path.
        """

        dxs = config.get("dxstream") or {}
        input_size = int(dxs.get("input_size", 640))

        self.bridge = PydxsBridge()
        self._native_frame_count = 0
        self._native_fps_t0: Optional[float] = None

        # Preflight: the native backend has NO software fallback. If the
        # dx_stream plugins / pydxs are not installed, fail loudly with an
        # actionable message instead of showing a silent black screen.
        missing = native_pipeline.missing_native_requirements(
            element_available=gst_element_available,
            pydxs_available=self.bridge.available,
        )
        if missing:
            raise RuntimeError(
                "This demo is dx_stream-only and requires dx_stream to be "
                "installed, but the following are missing:\n  - "
                + "\n  - ".join(missing)
                + "\n\nInstall dx_stream + pydxs on this machine and ensure the "
                "GStreamer plugins are on GST_PLUGIN_PATH (e.g. "
                "./install.sh or ./scripts/install_dxstream.sh)."
            )

        # Lightweight overlay metadata (no NPU load in Python).
        self.engine = NativeDisplayMeta(input_size, input_size)
        self._build_class_filter_panel(self.engine.classes)
        for w in self.video_widgets:
            w.set_overlay_style(self.engine.classes, self.engine.color_palette)

        pre_cfg, inf_cfg, post_cfg = native_config.build_native_cfgs(
            config, input_size, input_size
        )

        # Display downscale (RGA dxscale): keeps Qt handling small RGB tiles
        # regardless of the source resolution (essential for 4K inputs, helpful
        # for FHD). Enabled by default; size is configurable and disabled when
        # explicitly set to 0/false.
        display_size = self._resolve_display_size(dxs)

        # NV12->RGB colour conversion: prefer the RGA ``dxconvert`` element when
        # available (offloads the conversion from the CPU); fall back to the CPU
        # ``videoconvert`` otherwise. Can be forced off via config.
        color_convert = self._resolve_color_convert(dxs)

        # Pace output to the source's native frame rate for smooth playback and
        # to avoid wasting NPU/VPU work decoding faster than the video's real
        # fps. Enabled by default; set dxstream.sync_to_fps: false to run flat
        # out (max throughput / benchmarking).
        #
        # Pacing is done in Python (StreamPipeline._pace), NOT via the appsink's
        # clock sync: a clock-synced appsink (sync=true) stalls the gapless
        # SEGMENT-loop seek on the dx_stream pipeline for several seconds, which
        # is what produced the visible ~2s gap when a clip looped. Running the
        # appsink sync=false keeps looping seamless while Python backpressure
        # still throttles the pipeline to the native fps.
        sync_to_fps = bool(dxs.get("sync_to_fps", True))

        print(
            "[INFO] dxstream backend: color_convert={} display_size={} "
            "sync_to_fps={}".format(
                color_convert,
                f"{display_size[0]}x{display_size[1]}" if display_size else "native",
                sync_to_fps,
            ),
            flush=True,
        )

        for idx, ch_cfg in enumerate(config.get("channels", [])):
            if not ch_cfg.get("enabled", True):
                continue

            pipeline_str = native_pipeline.build_infer_pipeline(
                source_type=ch_cfg.get("type", "video"),
                source=ch_cfg["source"],
                preprocess_cfg=pre_cfg,
                infer_cfg=inf_cfg,
                postprocess_cfg=post_cfg,
                appsink_name=f"sink{idx}",
                display_size=display_size,
                color_convert=color_convert,
                sync=False,
            )
            pipe = StreamPipeline(
                channel_id=idx,
                pipeline_str=pipeline_str,
                bridge=self.bridge,
                frame_callback=self._on_native_frame,
                detection_callback=self._on_native_detections,
                appsink_name=f"sink{idx}",
                error_callback=self._on_native_error,
                meta_src_name=native_pipeline.meta_source_name(f"sink{idx}"),
                loop=ch_cfg.get("type", "video") == "video",
                source_size_callback=self._on_native_source_size,
                pace_fps=sync_to_fps,
            )
            self.stream_pipelines.append(pipe)

        for pipe in self.stream_pipelines:
            pipe.start()

    def _resolve_display_size(
        self, dxs: Dict[str, Any]
    ) -> Optional[tuple]:
        """Resolve the RGA display-downscale size from config.

        Defaults to 960x540 (a quarter of FHD, well-sized for one tile of a 2x2
        grid). Returns ``None`` when explicitly disabled so the full-resolution
        frame is delivered to Qt.
        """

        # Explicit per-axis override.
        if dxs.get("display_width") and dxs.get("display_height"):
            return (int(dxs["display_width"]), int(dxs["display_height"]))

        # Explicit disable: display_downscale: false (or 0).
        if "display_downscale" in dxs and not dxs.get("display_downscale"):
            return None

        return (960, 540)

    def _resolve_color_convert(self, dxs: Dict[str, Any]) -> str:
        """Pick the NV12->RGB backend ('rga' or 'cpu').

        Honours an explicit ``dxstream.color_convert`` config value; otherwise
        auto-selects the RGA ``dxconvert`` element when it is registered,
        falling back to the CPU ``videoconvert``.
        """

        requested = str(dxs.get("color_convert", "auto")).lower()
        if requested == "cpu":
            return "cpu"
        if requested == "rga":
            if not gst_element_available("dxconvert"):
                print(
                    "[WARN] dxstream.color_convert=rga requested but 'dxconvert' "
                    "is not available; falling back to CPU videoconvert.",
                    flush=True,
                )
                return "cpu"
            return "rga"
        # auto
        return "rga" if gst_element_available("dxconvert") else "cpu"

    def _on_native_frame(self, channel_id: int, frame: np.ndarray) -> None:
        """GStreamer-thread callback: forward a decoded frame to the Qt display."""

        if getattr(self, "_native_fps_t0", None) is None:
            self._native_fps_t0 = time.time()
        self._native_frame_count = getattr(self, "_native_frame_count", 0) + 1
        ch, out_frame, meta = native_signal.build_frame_payload(channel_id, frame)
        self.frame_ready.emit(ch, out_frame, meta)

    def _on_native_source_size(
        self, channel_id: int, width: int, height: int
    ) -> None:
        """GStreamer-thread callback: original detection-space resolution found."""

        self.source_size_ready.emit(channel_id, width, height)

    @QtCore.Slot(int, int, int)
    def on_source_size_ready(
        self, channel_id: int, width: int, height: int
    ) -> None:
        """Store the original detection resolution on the channel's widget."""

        if 0 <= channel_id < len(self.video_widgets):
            self.video_widgets[channel_id].set_detection_source_size(width, height)

    def _on_native_error(self, channel_id: int, message: str) -> None:
        """GStreamer-thread callback: a native pipeline reported a fatal error.

        StreamPipeline already logs the full error to stderr; this hook keeps a
        single place to react (currently informational) without crashing the
        GLib loop thread.
        """

        print(
            f"[ERROR] Channel {channel_id}: native pipeline error - {message}",
            file=sys.stderr,
            flush=True,
        )

    def _on_native_detections(
        self, channel_id: int, detections: np.ndarray
    ) -> None:
        """GStreamer-thread callback: filter + forward detections to the overlay."""

        detections = filter_by_classes(detections, self.get_selected_classes())
        ch, out_det, meta = native_signal.build_detection_payload(
            channel_id, detections
        )
        self.detections_ready.emit(ch, out_det, meta)

    def _on_paint_timer(self) -> None:
        """Called periodically to repaint all VideoWidgets.

        - Each VideoWidget draws the latest frame it holds.
        - Per-channel stats text is also updated here for use in the overlay.
        """

        for ch_id, w in enumerate(self.video_widgets):
            stats_text = self._get_channel_stats_snapshot(ch_id)
            w.update_stats_text(stats_text)
            w.update()

    # ----- GUI slots -----

    @QtCore.Slot(int, object, dict)
    def on_frame_ready(self, channel_id: int, frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Receive a captured frame (display path) and update the VideoWidget.

        - Runs at capture FPS; always applies only the latest frame per channel.
        - Per-channel display stats (fps/processed) are updated here so they
          reflect on-screen smoothness rather than inference throughput.
        """

        # Update the latest frame/meta per channel
        with self._latest_lock:
            self._latest_frames[channel_id] = frame
            self._latest_meta[channel_id] = meta

        # Update the widget using the latest frame at this moment
        if 0 <= channel_id < len(self.video_widgets):
            latest_frame = None
            latest_meta = None
            with self._latest_lock:
                latest_frame = self._latest_frames.get(channel_id)
                latest_meta = self._latest_meta.get(channel_id, {})

            if latest_frame is not None:
                color_format = (latest_meta or {}).get("color_format", "bgr")
                self.video_widgets[channel_id].set_frame(latest_frame, color_format)

            # Per-channel display stats (frames that reached the screen)
            self._update_channel_stats_on_frame(channel_id, latest_meta or {})

        # Print global throughput summary log once every 10 seconds from the main thread
        self._log_throughput_summary()

    @QtCore.Slot(int, object, dict)
    def on_detections_ready(
        self, channel_id: int, detections: np.ndarray, meta: Dict[str, Any]
    ) -> None:
        """Receive finalized detections and store them for paintEvent overlay."""

        if 0 <= channel_id < len(self.video_widgets):
            self.video_widgets[channel_id].set_detections(detections)

        # Inference-stage performance metrics come from the inference meta here.
        if meta is not None:
            self._update_performance_metrics(meta)

    # ----- Shutdown -----

    def _stop_stream_pipelines(self) -> None:
        """Request the native dx_stream pipelines to stop."""

        for pipe in getattr(self, "stream_pipelines", []):
            try:
                pipe.stop()
            except Exception as exc:  # pragma: no cover - board glue
                print(f"[WARN] stream pipeline stop failed: {exc}")

    def _cleanup_engine(self) -> None:
        """Release the display-metadata engine if it was created."""

        if hasattr(self, "engine") and self.engine:
            del self.engine

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover
        """Clean up the native pipelines when the window closes."""
        print("[INFO] Shutting down...")

        self._stop_stream_pipelines()
        self._cleanup_engine()

        print("[INFO] Shutdown complete")
        event.accept()


# ===== Config loading and app startup =====


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:  # pragma: no cover - entry point
    base_dir = Path(__file__).resolve().parent
    default_cfg = base_dir / "config" / "yolo26_multich.yaml"

    if not default_cfg.exists():
        print(f"[ERROR] Config file not found: {default_cfg}")
        sys.exit(1)

    config = load_config(str(default_cfg))

    # Pin the process (and all threads/GStreamer workers spawned afterwards) to
    # the configured CPU cluster, auto-detected from per-core max frequency.
    cpu_affinity.apply_affinity(str(config.get("cpu_affinity", "performance")))

    app = QtWidgets.QApplication(sys.argv)

    # Apply dark theme
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

    # Keep default style so scroll areas, checkboxes, etc. render well with the dark theme

    win = MainWindow(config)
    win.resize(1280, 720)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
