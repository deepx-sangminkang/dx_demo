"""YOLOv11 multi-channel Qt demo skeleton.

- Per-channel capture threads + global pre/infer/post workers + Qt GUI 2x2 layout
- Minimal skeleton code for understanding the overall structure and flow
- Real exception handling, shutdown handling, and detail work can be refined later
"""

from __future__ import annotations

import sys
import threading
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add the project root to sys.path so the script can run from anywhere
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from demo.engine import YOLOv11Engine
from demo.workers import (
    CaptureThread,
    preprocess_worker,
    wait_worker,
    postprocess_worker,
    draw_worker,
    queue_drop_counts,
    queue_drop_lock,
    get_fps,
    throughput_stats,
    throughput_lock,
)


# ===== Simple VideoWidget =====


class VideoWidget(QtWidgets.QWidget):
    """Widget that displays video for a single channel.

    - `set_frame(np.ndarray)` stores only the latest frame,
      and actual drawing happens in `paintEvent`.
    - This avoids excessive external repaint requests and enables smooth,
      timer-driven updates.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: black;")
        self._latest_frame: Optional[np.ndarray] = None
        # Text used to show per-channel statistics
        # (FPS, processed frame count, dropped frame count)
        self._stats_text: str = ""

    @QtCore.Slot(np.ndarray)
    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Store a BGR image in the internal buffer."""

        if frame_bgr is None:
            return

        self._latest_frame = frame_bgr

    def update_stats_text(self, text: str) -> None:
        """Store external stats text for use in the overlay.

        - It is drawn on top of the video in `paintEvent`.
        """

        self._stats_text = text

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - GUI only
        painter = QtGui.QPainter(self)
        rect = self.rect()

        if self._latest_frame is not None:
            frame_rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )
            pix = QtGui.QPixmap.fromImage(qimg)

            # Draw while preserving aspect ratio inside the widget size
            target = pix.scaled(
                rect.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - target.width()) // 2
            y = rect.y() + (rect.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)
        else:
            # Fill only the background
            painter.fillRect(rect, self.palette().window())

        # Draw the stats text overlay on top of the video
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


# ===== Main window =====


class MainWindow(QtWidgets.QMainWindow):
    frame_ready = QtCore.Signal(int, object, dict)  # channel_id, frame_bgr, meta

    def __init__(self, config: Dict[str, Any], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("YOLOv11 Multi-Channel Demo")
        # Set of selected class IDs (written only from the GUI thread, read by workers)
        self._selected_classes: Set[int] = set()
        self._selected_lock = threading.Lock()

        # Accumulator for performance metrics
        self._metrics: Dict[str, Any] = {
            "frames": 0,
            "sum_read": 0.0,
            "sum_pre": 0.0,
            "sum_inf": 0.0,
            "sum_post": 0.0,
            "sum_post_engine": 0.0,
            "sum_post_filter": 0.0,
            "sum_post_draw": 0.0,
            "start_ts": None,
        }
        self._metrics_lock = threading.Lock()

    # Storage for per-channel statistics (processed/FPS)
    # - dropped_* values are read directly from workers.queue_drop_counts.
        self._channel_stats: Dict[int, Dict[str, Any]] = {}
        self._channel_stats_lock = threading.Lock()

        # Buffers that temporarily hold the latest frame per channel
        # - Even if frame_ready signals arrive in bursts from workers,
        #   the GUI uses only the most recent frame per channel.
        self._latest_frames: Dict[int, np.ndarray] = {}
        self._latest_meta: Dict[int, Dict[str, Any]] = {}
        self._latest_lock = threading.Lock()

        # Timer for refreshing the screen at a fixed cadence (for example, about 30 FPS)
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.timeout.connect(self._on_paint_timer)
        self._paint_timer.start(33)  # 33 ms interval ~= 30 FPS

        # Build the central 2x2 layout
        central = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(central)
        self.video_widgets: List[VideoWidget] = []
        for i in range(4):
            w = VideoWidget()
            self.video_widgets.append(w)
            row, col = divmod(i, 2)
            grid.addWidget(w, row, col)
        self.setCentralWidget(central)

        # Class filter panel on the right (starts empty and is filled after engine creation)
        self.class_dock = QtWidgets.QDockWidget("", self)
        self.class_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        # Do not use the title-bar close(X)/float buttons;
        # control the panel only through the side toggle button on the left.
        self.class_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        # Default width range while the panel is open (limited to avoid breaking layout)
        self._dock_min_width_expanded = 220
        self._dock_max_width_expanded = 320
        self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
        self.class_dock.setMaximumWidth(self._dock_max_width_expanded)

        # Container with the toggle button on the left and the real class panel on the right
        self.dock_outer_widget = QtWidgets.QWidget(self.class_dock)
        outer_layout = QtWidgets.QHBoxLayout(self.dock_outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        # Round toggle button placed mid-left (always visible even when the panel collapses)
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

        # Use a fixed size policy so the toggle button occupies as little panel width as possible
        self.side_toggle_button.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )

        outer_layout.addWidget(self.side_toggle_button, 0, QtCore.Qt.AlignVCenter)

        # Widget containing the actual class-panel contents (only this part is hidden on toggle)
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

        # Area that contains the actual checkbox list (inside the scroll area)
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

        # Connect the frame_ready signal to the VideoWidget update slot
        self.frame_ready.connect(self.on_frame_ready)

        # Sync side-button state when the dock widget visibility changes
        self.class_dock.visibilityChanged.connect(self._on_class_dock_visibility_changed)

        # Initialise the engine and worker/queue pipeline
        self._init_engine_and_workers(config)

    # ----- Class filter -----

    def _build_class_filter_panel(self, class_names: List[str]) -> None:
        """Build the class-name list as checkboxes.

        - With COCO80 there are roughly 80 entries, so a scrollable layout works well.
        """

        # Clear existing checkboxes (keep the stretch, remove only widgets)
        for i in reversed(range(self.class_list_layout.count())):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setParent(None)

        # Fill actual checkboxes from the top
        for class_id, name in enumerate(class_names):
            cb = QtWidgets.QCheckBox(f"{class_id}: {name}")
            cb.setChecked(True)  # Show all classes by default
            cb.stateChanged.connect(self._on_any_class_checkbox_changed)

            self.class_list_layout.insertWidget(
                self.class_list_layout.count() - 1, cb
            )

        # Build the selected-class set once using the initial all-checked state
        self._rebuild_selected_classes()

    def _on_select_all(self) -> None:
        """Mark all classes as selected."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(True)

    def _on_clear_all(self) -> None:
        """Clear the selection state for all classes."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(False)

    def _on_any_class_checkbox_changed(self, state: int) -> None:
        """Rebuild the selected-class set whenever any class checkbox changes."""

        self._rebuild_selected_classes()

    def _rebuild_selected_classes(self) -> None:
        """Rebuild `_selected_classes` from the current checkbox states."""

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

    # ----- Performance metrics / logging -----

    def _update_performance_metrics(self, meta: Dict[str, Any]) -> None:
        """Accumulate performance metrics from per-frame metadata and emit periodic summary logs."""

        t_read = meta.get("t_read", 0.0)
        t_pre = meta.get("t_preprocess", 0.0)
        t_inf = meta.get("t_inference", 0.0)
        # The post stage is measured as post(engine) + post(filter); draw is tracked separately.
        t_post = meta.get("t_postprocess", 0.0)
        t_pp_engine = meta.get("t_post_engine", 0.0)
        t_pp_filter = meta.get("t_post_filter", 0.0)
        t_pp_draw = meta.get("t_post_draw", 0.0)

        now = time.perf_counter()

        with self._metrics_lock:
            m = self._metrics

            if m["start_ts"] is None:
                m["start_ts"] = now

            m["frames"] += 1
            m["sum_read"] += t_read
            m["sum_pre"] += t_pre
            m["sum_inf"] += t_inf
            m["sum_post"] += t_post
            m["sum_post_engine"] += t_pp_engine
            m["sum_post_filter"] += t_pp_filter
            m["sum_post_draw"] += t_pp_draw

            # Old per-frame average logs were replaced by throughput logs,
            # so only the accumulated statistics are kept here.

    def _log_throughput_summary(self) -> None:
        """Log per-stage throughput every 10 seconds based on global `throughput_stats`."""

        now = time.time()
        # If no previous log time exists, initialise it and return
        last = getattr(self, "_last_throughput_log_ts", None)
        if last is None:
            self._last_throughput_log_ts = now
            return

        if now - last < 10.0:
            return

        self._last_throughput_log_ts = now

        # Read throughput statistics from the workers module
        read_fps = get_fps("read")
        pre_fps = get_fps("pre")
        inf_fps = get_fps("inf")
        post_fps = get_fps("post")
        draw_fps = get_fps("draw")

        # Overall FPS is based on frames completed at the draw stage
        with throughput_lock:
            draw_stats = throughput_stats["draw"]
            d_first = draw_stats["first_ts"]
            d_last = draw_stats["last_ts"]
            d_count = draw_stats["count"]

        if d_first is not None and d_last is not None and d_last > d_first and d_count > 0:
            overall_fps = d_count / (d_last - d_first)
        else:
            overall_fps = 0.0

        print(
            "[THROUGHPUT] read={:.1f} fps, pre={:.1f} fps, inf={:.1f} fps, post={:.1f} fps, draw={:.1f} fps, overall={:.1f} fps".format(
                read_fps,
                pre_fps,
                inf_fps,
                post_fps,
                draw_fps,
                overall_fps,
            )
        )

    # ----- Per-channel stats updates -----

    def _update_channel_stats_on_frame(self, channel_id: int, meta: Dict[str, Any]) -> None:
        """Update per-channel stats using frames that reached `on_frame_ready`.

        - `processed_frames`: number of frames that actually reached the screen
        - `fps`: computed from processed frame increments over the last second
        - drop counts come from `workers.queue_drop_counts`, not from `meta`
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

            # Update FPS using elapsed time and frame count since the last checkpoint
            elapsed = now - s["last_ts"]
            if elapsed >= 1.0:
                delta_frames = s["processed_frames"] - s["last_frames_for_fps"]
                s["fps"] = delta_frames / elapsed if elapsed > 0 else 0.0
                s["last_ts"] = now
                s["last_frames_for_fps"] = s["processed_frames"]

    def _get_channel_stats_snapshot(self, channel_id: int) -> str:
        """Return current per-channel stats as a compact string.

    Example: "ch1 fps=29.8 proc=1234 drop(input=10 infer=5 post=3 draw=1)"
        """

        with self._channel_stats_lock:
            s = self._channel_stats.get(channel_id)
            if not s:
                fps = 0.0
                proc = 0
            else:
                fps = s["fps"]
                proc = s["processed_frames"]

        # Read per-queue drop counts directly from workers.queue_drop_counts
        with queue_drop_lock:
            drop_input = queue_drop_counts["input"].get(channel_id, 0)
            drop_infer = queue_drop_counts["infer"].get(channel_id, 0)
            drop_post = queue_drop_counts["post"].get(channel_id, 0)
            drop_draw = queue_drop_counts["draw"].get(channel_id, 0)

        return (
            f"ch{channel_id+1} "
            f"fps={fps:.1f} "
            f"proc={proc} "
                f"drop(input={drop_input} infer={drop_infer} post={drop_post} draw={drop_draw})"
        )

    # ----- Class panel toggle -----

    def _on_side_toggle_clicked(self, checked: bool) -> None:
        """Show or hide only the class-panel contents when the left round button is clicked.

        Keep the dock itself visible and hide only the class-panel content
        so the toggle button never disappears.
        """

        self.class_panel_widget.setVisible(checked)

        # When the panel collapses, shrink the dock to roughly the button width;
        # restore the original width range when it expands again.
        if checked:
            # Expanded state: restore the original configured width range.
            self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
            self.class_dock.setMaximumWidth(self._dock_max_width_expanded)
        else:
            # Collapsed state: clamp width tightly so only the toggle button stays visible.
            collapsed_width = self.side_toggle_button.width() + 8
            self.class_dock.setMinimumWidth(collapsed_width)
            self.class_dock.setMaximumWidth(collapsed_width)

        # In expanded state (`checked=True`) use '>' (feels like folding right);
        # in hidden state use '<' (feels like expanding left).
        self.side_toggle_button.setText(">" if checked else "<")

    def _on_class_dock_visibility_changed(self, visible: bool) -> None:
        """Keep the button state reasonable even when the entire dock is hidden.

        If the dock is fully closed (for example, when the user presses X), it may need to be shown again,
        so this keeps the button text consistent without touching the checked state.
        """

        # Even when the dock is hidden, the button reappears together with it when reopened,
        # so no special sync is required beyond keeping the text consistent.
        if visible and self.class_panel_widget.isVisible():
            self.side_toggle_button.setText(">")
        elif visible:
            self.side_toggle_button.setText("<")

    def get_selected_classes(self) -> Set[int]:
        """Return the selected class set for worker-side reads."""

        with self._selected_lock:
            return set(self._selected_classes)

    # ----- Engine and worker initialisation -----

    def _init_engine_and_workers(self, config: Dict[str, Any]) -> None:
        """Initialise the YOLO engine, queues, and threads (capture + global workers)."""

        model_path = config["model"]
        self.engine = YOLOv11Engine(model_path)

        # Build the filter panel from class names
        self._build_class_filter_panel(self.engine.classes)

        # Shared queues and stop flag
        self.input_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        self.infer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        # Queue for results passed from wait -> postprocess
        self.post_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        # Queue for results passed from postprocess -> draw
        self.draw_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()

        # Create per-channel capture threads
        self.capture_threads: List[CaptureThread] = []
        for idx, ch_cfg in enumerate(config.get("channels", [])):
            if not ch_cfg.get("enabled", True):
                continue

            source = ch_cfg["source"]
            max_fps = ch_cfg.get("max_fps")
            t = CaptureThread(
                channel_id=idx,
                source=source,
                input_queue=self.input_queue,
                max_fps=max_fps,
            )
            self.capture_threads.append(t)

        # Create global worker threads with threading.Thread
        self._threads: List[threading.Thread] = []

        # Read per-stage worker counts from the YAML workers section
        workers_cfg = config.get("workers", {})
        if not isinstance(workers_cfg, dict):
            workers_cfg = {}

        def _get_worker_count(key: str, default: int) -> int:
            """Safely read an integer value from the workers section."""

            try:
                value = int(workers_cfg.get(key, default))
            except (TypeError, ValueError):
                value = default
            # Clamp values below 1 up to a minimum of 1
            return max(1, value)

        num_pre_workers = _get_worker_count("preprocess", 1)
        num_wait_workers = _get_worker_count("wait", 1)
        num_post_workers = _get_worker_count("postprocess", 1)
        num_draw_workers = _get_worker_count("draw", 1)

        # Create the preprocess worker pool
        for i in range(num_pre_workers):
            t_pre = threading.Thread(
                target=preprocess_worker,
                args=(self.engine, self.input_queue, self.infer_queue, self.stop_event),
                daemon=True,
                name=f"preprocess_worker_{i}",
            )
            self._threads.append(t_pre)

        # Create the wait worker pool
        for i in range(num_wait_workers):
            t_wait = threading.Thread(
                target=wait_worker,
                args=(self.engine, self.infer_queue, self.post_queue, self.stop_event),
                daemon=True,
                name=f"wait_worker_{i}",
            )
            self._threads.append(t_wait)

        # Create the postprocess worker pool (post_queue -> draw_queue)
        for i in range(num_post_workers):
            t_post = threading.Thread(
                target=postprocess_worker,
                args=(
                    self.engine,
                    self.post_queue,
                    self.draw_queue,
                    self.get_selected_classes,
                    self.stop_event,
                ),
                daemon=True,
                name=f"postprocess_worker_{i}",
            )
            self._threads.append(t_post)

        # Create the draw worker pool (draw_queue -> GUI)
        for i in range(num_draw_workers):
            t_draw = threading.Thread(
                target=draw_worker,
                args=(
                    self.engine,
                    self.draw_queue,
                    self._on_frame_ready_from_worker,
                    self.stop_event,
                ),
                daemon=True,
                name=f"draw_worker_{i}",
            )
            self._threads.append(t_draw)

        # Start all threads
        for t in self.capture_threads:
            t.start()
        for t in self._threads:
            t.start()

    # ----- Worker -> GUI forwarding wrapper -----

    def _on_frame_ready_from_worker(
        self, channel_id: int, frame_bgr: np.ndarray, meta: Dict[str, Any]
    ) -> None:
        """Callback invoked from a worker thread.

        - Wrapped so it emits a signal to the Qt main thread.
        """

        # This is called from a worker thread, so just emit the signal here.
        # The Qt main thread uses only the latest frame in on_frame_ready.
        self.frame_ready.emit(channel_id, frame_bgr, meta)

    def _on_paint_timer(self) -> None:
        """Periodically repaint all VideoWidgets.

        - Each VideoWidget draws the latest frame it holds.
        - Per-channel stats text is also refreshed here for overlay use.
        """

        for ch_id, w in enumerate(self.video_widgets):
            stats_text = self._get_channel_stats_snapshot(ch_id)
            w.update_stats_text(stats_text)
            w.update()

    # ----- GUI slots -----

    @QtCore.Slot(int, object, dict)
    def on_frame_ready(self, channel_id: int, frame_bgr: np.ndarray, meta: Dict[str, Any]) -> None:
        """Handle the frame_ready signal and update the VideoWidget.

        - Even if many frame_ready signals arrive in a burst,
          only the latest frame per channel is applied to the screen.
        """

        # Refresh the latest frame/meta for this channel
        with self._latest_lock:
            self._latest_frames[channel_id] = frame_bgr
            self._latest_meta[channel_id] = meta

        # Update the widget using the latest frame available at this moment
        if 0 <= channel_id < len(self.video_widgets):
            latest_frame = None
            latest_meta = None
            with self._latest_lock:
                latest_frame = self._latest_frames.get(channel_id)
                latest_meta = self._latest_meta.get(channel_id, {})

            if latest_frame is not None:
                self.video_widgets[channel_id].set_frame(latest_frame)

            # Update performance metrics and per-channel stats
            if latest_meta is not None:
                self._update_performance_metrics(latest_meta)
                self._update_channel_stats_on_frame(channel_id, latest_meta)

        # Emit the global throughput summary once every 10 seconds from the main thread
        self._log_throughput_summary()

    # ----- Shutdown -----

    def _join_alive_threads(self, threads: List[Any], timeout: float) -> None:
        """Join only threads that still exist and are running."""

        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)

    def _stop_capture_threads(self) -> None:
        """Request capture threads to stop and wait briefly for exit."""

        for thread in self.capture_threads:
            if thread:
                thread.stop()

        self._join_alive_threads(self.capture_threads, timeout=2.0)

    def _worker_queues(self) -> List["queue.Queue[Any]"]:
        """Return queues that need sentinel values during shutdown."""

        return [self.input_queue, self.infer_queue, self.post_queue, self.draw_queue]

    def _stop_worker_threads(self) -> None:
        """Push sentinel values and wait for worker threads to exit."""

        threads = getattr(self, "_threads", [])
        for _ in range(len(threads)):
            for work_queue in self._worker_queues():
                try:
                    work_queue.put_nowait(None)
                except queue.Full:
                    pass

        self._join_alive_threads(threads, timeout=1.0)

    def _cleanup_engine(self) -> None:
        """Release the engine if it was created."""

        if hasattr(self, "engine") and self.engine:
            del self.engine

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover
        """Clean up workers and capture threads when the window closes."""
        print("[INFO] Shutdown started...")

        self.stop_event.set()

        self._stop_capture_threads()
        self._stop_worker_threads()
        self._cleanup_engine()

        print("[INFO] Shutdown complete")
        event.accept()


# ===== Config loading and app startup =====


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:  # pragma: no cover - entry point
    base_dir = Path(__file__).resolve().parent
    default_cfg = base_dir / "config" / "yolov11_multich.yaml"

    if not default_cfg.exists():
        print(f"[ERROR] Config file not found: {default_cfg}")
        sys.exit(1)

    config = load_config(str(default_cfg))

    # cv2.setNumThreads(1)

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

    # Keep default widget styling so scroll areas, checkboxes, and similar controls fit the dark theme

    win = MainWindow(config)
    win.resize(1280, 720)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
