"""YOLO26 멀티 채널 Qt 데모 스켈레톤.

- 채널별 캡처 스레드 + 전역 pre/infer/draw 워커 + Qt GUI 2x2 레이아웃
- 구조와 흐름을 이해하기 위한 최소한의 뼈대 코드
- 실제 예외 처리 / 종료 처리 / 세부 기능은 이후 단계에서 보완
"""

from __future__ import annotations

import sys
import threading
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 프로젝트 루트를 sys.path에 추가 (어디서든 실행 가능하도록)
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from demo.engine import YOLO26Engine
from demo.workers import (
    CaptureThread,
    preprocess_worker,
    wait_worker,
    draw_worker,
    queue_drop_counts,
    queue_drop_lock,
    get_fps,
    throughput_stats,
    throughput_lock,
)


# ===== 간단한 VideoWidget =====


class VideoWidget(QtWidgets.QWidget):
    """각 채널의 영상을 보여주는 위젯.

    - set_frame(np.ndarray)는 최신 프레임만 보관하고,
      실제 그리기는 paintEvent 에서 수행한다.
    - 이렇게 하면 외부에서 너무 자주 repaint 를 요청하지 않고,
      타이머 기반으로 부드럽게 갱신할 수 있다.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: black;")
        self._latest_frame: Optional[np.ndarray] = None
        # 채널별 통계를 표시하기 위한 텍스트 정보
        # (FPS, 처리된 프레임 수, 드롭된 프레임 수)
        self._stats_text: str = ""

    @QtCore.Slot(np.ndarray)
    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """BGR 이미지를 받아 내부 버퍼에 저장만 한다."""

        if frame_bgr is None:
            return

        self._latest_frame = frame_bgr

    def update_stats_text(self, text: str) -> None:
        """외부에서 전달한 통계 문자열을 overlay 용으로 저장.

        - paintEvent 에서 영상 위에 함께 그린다.
        """

        self._stats_text = text

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - GUI 전용
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

            # 위젯 크기에 맞게 비율 유지하며 그리기
            target = pix.scaled(
                rect.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - target.width()) // 2
            y = rect.y() + (rect.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)
        else:
            # 배경만 채우기
            painter.fillRect(rect, self.palette().window())

        # 영상 위에 통계 텍스트 overlay
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


# ===== 메인 윈도우 =====


class MainWindow(QtWidgets.QMainWindow):
    frame_ready = QtCore.Signal(int, object, dict)  # channel_id, frame_bgr, meta

    def __init__(self, config: Dict[str, Any], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("YOLO26 Multi-Channel Demo")
        # 선택된 클래스 ID 집합 (GUI 스레드에서만 쓰기, 워커는 읽기만)
        self._selected_classes: Set[int] = set()
        self._selected_lock = threading.Lock()

        # 성능 측정을 위한 메트릭 누적용 구조체
        self._metrics: Dict[str, Any] = {
            "frames": 0,
            "sum_read": 0.0,
            "sum_pre": 0.0,
            "sum_inf": 0.0,
            "sum_draw": 0.0,
            "start_ts": None,
        }
        self._metrics_lock = threading.Lock()

    # 채널별 통계 (처리/FPS) 저장용 구조체
    # - dropped_* 값은 workers.queue_drop_counts 를 직접 읽어와 사용한다.
        self._channel_stats: Dict[int, Dict[str, Any]] = {}
        self._channel_stats_lock = threading.Lock()

        # 채널별로 가장 최신 프레임을 잠깐 보관하기 위한 버퍼
        # - 워커에서 frame_ready 시그널이 몰려와도,
        #   GUI 는 각 채널의 최신 프레임만 사용하도록 한다.
        self._latest_frames: Dict[int, np.ndarray] = {}
        self._latest_meta: Dict[int, Dict[str, Any]] = {}
        self._latest_lock = threading.Lock()

        # 화면 갱신을 일정한 주기로 수행하기 위한 타이머 (예: 약 30 FPS)
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.timeout.connect(self._on_paint_timer)
        self._paint_timer.start(33)  # 33ms 간격 ≒ 30 FPS

        # 중앙 2x2 레이아웃 구성
        central = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(central)
        self.video_widgets: List[VideoWidget] = []
        for i in range(4):
            w = VideoWidget()
            self.video_widgets.append(w)
            row, col = divmod(i, 2)
            grid.addWidget(w, row, col)
        self.setCentralWidget(central)

        # 우측 클래스 필터 패널 (기본은 비어있고, 엔진 생성 후 채울 예정)
        self.class_dock = QtWidgets.QDockWidget("", self)
        self.class_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        # 상단 제목줄의 닫기(X)/플로트 버튼은 사용하지 않고,
        # 좌측의 사이드 토글 버튼으로만 패널을 제어한다.
        self.class_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        # 패널 열려 있을 때의 기본 폭 범위 (레이아웃이 깨지지 않도록 제한).
        self._dock_min_width_expanded = 220
        self._dock_max_width_expanded = 320
        self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
        self.class_dock.setMaximumWidth(self._dock_max_width_expanded)

        # 좌측에 토글 버튼 + 우측에 실제 클래스 패널이 들어갈 컨테이너
        self.dock_outer_widget = QtWidgets.QWidget(self.class_dock)
        outer_layout = QtWidgets.QHBoxLayout(self.dock_outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        # 좌측 중간에 위치할 동그란 토글 버튼 (도크 밖에서도 항상 보임)
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

        # 토글 버튼이 패널 폭을 거의 차지하지 않도록, 고정 폭 + 고정 크기 정책을 사용
        self.side_toggle_button.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )

        outer_layout.addWidget(self.side_toggle_button, 0, QtCore.Qt.AlignVCenter)

        # 실제 클래스 패널 내용 위젯 (토글 시 이 부분만 숨김)
        self.class_panel_widget = QtWidgets.QWidget(self.dock_outer_widget)
        self.class_panel_layout = QtWidgets.QVBoxLayout(self.class_panel_widget)
        self.class_panel_layout.setContentsMargins(4, 4, 4, 4)
        self.class_panel_layout.setSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear_all)
        btn_row.addStretch(1)

        self.class_panel_layout.addLayout(btn_row)

        # 실제 체크박스 리스트가 들어갈 영역 (스크롤 영역의 컨테이너)
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

        # frame_ready 시그널을 VideoWidget 업데이트 슬롯에 연결
        self.frame_ready.connect(self.on_frame_ready)

        # 도크 위젯의 보이기/숨기기 상태가 바뀔 때 사이드 버튼 상태 동기화
        self.class_dock.visibilityChanged.connect(self._on_class_dock_visibility_changed)

        # 엔진 및 워커/큐 초기화
        self._init_engine_and_workers(config)

    # ----- 클래스 필터 관련 -----

    def _build_class_filter_panel(self, class_names: List[str]) -> None:
        """class name 리스트를 체크박스로 생성.

        - COCO80 기준이면 80개 정도, 스크롤이 가능하도록 구성하는 것이 좋다.
        """

        # 기존 체크박스 비우기 (스트레치는 남기고 위젯만 제거)
        for i in reversed(range(self.class_list_layout.count())):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setParent(None)

        # 실제 체크박스들을 위쪽부터 채우기
        for class_id, name in enumerate(class_names):
            cb = QtWidgets.QCheckBox(f"{class_id}: {name}")
            cb.setChecked(True)  # 기본은 모두 표시
            cb.stateChanged.connect(self._on_any_class_checkbox_changed)

            self.class_list_layout.insertWidget(
                self.class_list_layout.count() - 1, cb
            )

        # 전체 선택/해제 버튼 동작 연결 (중복 연결 방지 위해 먼저 기존 연결 해제 시도)
        try:
            self.btn_select_all.clicked.disconnect()
        except TypeError:
            # 연결이 없을 때 disconnect 하면 TypeError 가 날 수 있음
            pass
        try:
            self.btn_clear_all.clicked.disconnect()
        except TypeError:
            pass

        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_clear_all.clicked.connect(self._on_clear_all)

        # 초기 상태(모두 체크)를 기준으로 선택된 클래스 집합을 한 번 빌드
        self._rebuild_selected_classes()

    def _on_select_all(self) -> None:
        """모든 클래스를 선택 상태로 변경."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(True)

    def _on_clear_all(self) -> None:
        """모든 클래스를 해제 상태로 변경."""
        for i in range(self.class_list_layout.count()):
            item = self.class_list_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(False)

    def _on_any_class_checkbox_changed(self, state: int) -> None:
        """어떤 클래스 체크박스든 상태가 바뀌면 전체를 다시 스캔해 집합을 재구성."""

        self._rebuild_selected_classes()

    def _rebuild_selected_classes(self) -> None:
        """체크박스들의 현재 상태를 기준으로 _selected_classes 를 다시 만든다."""

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

    # ----- 성능 측정/로그 -----

    def _update_performance_metrics(self, meta: Dict[str, Any]) -> None:
        """프레임별 메타 정보로부터 성능 메트릭을 누적하고, 주기적으로 요약 로그를 출력."""

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

            # 기존 per-frame 평균 지표 로그는 throughput 로그로 대체되므로
            # 여기서는 누적 통계만 유지하고 별도 출력은 하지 않는다.

    def _log_throughput_summary(self) -> None:
        """전역 throughput_stats 를 기반으로 10초 간격으로 단계별 처리량을 로그로 출력."""

        now = time.time()
        # 마지막 로그 시각이 없으면 초기화만 하고 리턴
        last = getattr(self, "_last_throughput_log_ts", None)
        if last is None:
            self._last_throughput_log_ts = now
            return

        if now - last < 10.0:
            return

        self._last_throughput_log_ts = now

        # workers 모듈의 처리량 통계를 읽어온다.
        read_fps = get_fps("read")
        pre_fps = get_fps("pre")
        inf_fps = get_fps("inf")
        draw_fps = get_fps("draw")

        # overall FPS 는 draw 단계에서 처리 완료된 프레임 기준
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
            "[THROUGHPUT] read={:.1f} fps, pre={:.1f} fps, inf={:.1f} fps, draw={:.1f} fps, overall={:.1f} fps".format(
                read_fps,
                pre_fps,
                inf_fps,
                draw_fps,
                overall_fps,
            )
        )

    # ----- 채널별 통계 업데이트 -----

    def _update_channel_stats_on_frame(self, channel_id: int, meta: Dict[str, Any]) -> None:
        """on_frame_ready 에 도달한 프레임 기준으로 per-channel 통계를 업데이트.

        - processed_frames: 실제 화면까지 도달한 프레임 수
        - fps: 최근 1초 동안 processed_frames 증가량으로 계산
        - 드롭 수는 workers.queue_drop_counts 를 사용 (meta 에서 가져오지 않음)
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

            # FPS 계산: 마지막 기준점 이후 경과 시간과 프레임 수로 갱신
            elapsed = now - s["last_ts"]
            if elapsed >= 1.0:
                delta_frames = s["processed_frames"] - s["last_frames_for_fps"]
                s["fps"] = delta_frames / elapsed if elapsed > 0 else 0.0
                s["last_ts"] = now
                s["last_frames_for_fps"] = s["processed_frames"]

    def _get_channel_stats_snapshot(self, channel_id: int) -> str:
        """현재까지의 채널별 통계를 간단한 문자열로 반환.

    예: "ch1 fps=29.8 proc=1234 drop(input=10 infer=5 draw=1)"
        """

        with self._channel_stats_lock:
            s = self._channel_stats.get(channel_id)
            if not s:
                fps = 0.0
                proc = 0
            else:
                fps = s["fps"]
                proc = s["processed_frames"]

        # 큐별 드롭 카운트는 workers.queue_drop_counts 에서 직접 읽어온다.
        with queue_drop_lock:
            drop_input = queue_drop_counts["input"].get(channel_id, 0)
            drop_infer = queue_drop_counts["infer"].get(channel_id, 0)
            drop_draw = queue_drop_counts["draw"].get(channel_id, 0)

        return (
            f"ch{channel_id+1} "
            f"fps={fps:.1f} "
            f"proc={proc} "
                f"drop(input={drop_input} infer={drop_infer} draw={drop_draw})"
        )

    # ----- 클래스 패널 토글 -----

    def _on_side_toggle_clicked(self, checked: bool) -> None:
        """좌측 둥근 버튼 클릭 시 클래스 패널 내용만 보이기/숨기기.

        도크 자체는 항상 보이게 두고, 클래스 패널 부분만 숨겨서
        토글 버튼이 사라지지 않도록 한다.
        """

        self.class_panel_widget.setVisible(checked)

        # 패널이 접힐 때는 도크 폭을 토글 버튼 폭 정도로 줄이고,
        # 다시 펼칠 때는 원래 폭 범위를 복원한다.
        if checked:
            # 펼쳐진 상태: 원래 설정한 폭 범위를 되살린다.
            self.class_dock.setMinimumWidth(self._dock_min_width_expanded)
            self.class_dock.setMaximumWidth(self._dock_max_width_expanded)
        else:
            # 접힌 상태: 토글 버튼만 보일 정도로 폭을 강하게 제한.
            collapsed_width = self.side_toggle_button.width() + 8
            self.class_dock.setMinimumWidth(collapsed_width)
            self.class_dock.setMaximumWidth(collapsed_width)

        # 펼쳐진 상태(checked=True)에서는 '>' (오른쪽으로 접히는 느낌),
        # 숨김 상태에서는 '<' (왼쪽으로 펼쳐지는 느낌)
        self.side_toggle_button.setText(">" if checked else "<")

    def _on_class_dock_visibility_changed(self, visible: bool) -> None:
        """도크 전체가 숨겨질 때도 버튼 상태를 적당히 유지.

        도크가 완전히 닫히면(사용자가 X를 누른 경우) 다시 열려야 하므로,
        여기서는 side_toggle_button 의 checked 상태는 건드리지 않고 텍스트만 정리한다.
        """

        # 도크가 숨겨져도 버튼은 다시 도크가 열릴 때 함께 나타나므로,
        # 특별한 추가 동기화는 필요하지 않지만, 일관성을 위해 텍스트만 맞춰준다.
        if visible and self.class_panel_widget.isVisible():
            self.side_toggle_button.setText(">")
        elif visible:
            self.side_toggle_button.setText("<")

    def get_selected_classes(self) -> Set[int]:
        """워커에서 읽어갈 수 있도록 선택된 클래스 집합을 반환."""

        with self._selected_lock:
            return set(self._selected_classes)

    # ----- 엔진 및 워커 초기화 -----

    def _init_engine_and_workers(self, config: Dict[str, Any]) -> None:
        """YOLO 엔진, 큐, 스레드(캡처 + 전역 워커)를 초기화."""

        model_path = config["model"]
        self.engine = YOLO26Engine(model_path)

        # 클래스 이름을 이용해 필터 패널 생성
        self._build_class_filter_panel(self.engine.classes)

        # 공용 큐 및 stop 플래그
        self.input_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        self.infer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        self.draw_queue: "queue.Queue[Any]" = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()

        # 채널별 캡처 스레드 생성
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

        # 전역 워커 스레드 생성 (threading.Thread 사용)
        self._threads: List[threading.Thread] = []

        # YAML 의 workers 설정에서 단계별 워커 수를 읽어온다.
        workers_cfg = config.get("workers", {})
        if not isinstance(workers_cfg, dict):
            workers_cfg = {}

        def _get_worker_count(key: str, default: int) -> int:
            """workers 섹션에서 정수 값을 안전하게 읽어오는 헬퍼."""

            try:
                value = int(workers_cfg.get(key, default))
            except (TypeError, ValueError):
                value = default
            # 1 미만 값이 들어오면 최소 1 개로 보정
            return max(1, value)

        num_pre_workers = _get_worker_count("preprocess", 1)
        num_wait_workers = _get_worker_count("wait", 1)
        num_draw_workers = _get_worker_count("draw", 1)

        # preprocess 워커 풀 생성
        for i in range(num_pre_workers):
            t_pre = threading.Thread(
                target=preprocess_worker,
                args=(self.engine, self.input_queue, self.infer_queue, self.stop_event),
                daemon=True,
                name=f"preprocess_worker_{i}",
            )
            self._threads.append(t_pre)

        # wait 워커 풀 생성
        for i in range(num_wait_workers):
            t_wait = threading.Thread(
                target=wait_worker,
                args=(self.engine, self.infer_queue, self.draw_queue, self.stop_event),
                daemon=True,
                name=f"wait_worker_{i}",
            )
            self._threads.append(t_wait)

        # draw 워커 풀 생성 (draw_queue → GUI)
        for i in range(num_draw_workers):
            t_draw = threading.Thread(
                target=draw_worker,
                args=(
                    self.engine,
                    self.draw_queue,
                    self.get_selected_classes,
                    self._on_frame_ready_from_worker,
                    self.stop_event,
                ),
                daemon=True,
                name=f"draw_worker_{i}",
            )
            self._threads.append(t_draw)

        # 스레드 시작
        for t in self.capture_threads:
            t.start()
        for t in self._threads:
            t.start()

    # ----- 워커 → GUI 전달 래핑 -----

    def _on_frame_ready_from_worker(
        self, channel_id: int, frame_bgr: np.ndarray, meta: Dict[str, Any]
    ) -> None:
        """워커 스레드에서 호출되는 콜백.

        - Qt 메인 스레드로 시그널을 emit 하도록 래핑.
        """

        # 워커 스레드에서 호출되므로, 여기서는 단순히 시그널만 emit 한다.
        # Qt 메인 스레드 쪽에서 최신 프레임만 사용하도록 on_frame_ready 에서 처리한다.
        self.frame_ready.emit(channel_id, frame_bgr, meta)

    def _on_paint_timer(self) -> None:
        """주기적으로 호출되어 모든 VideoWidget 을 repaint 한다.

        - 각 VideoWidget 은 자신이 보관하고 있는 최신 프레임을 그린다.
        - 이 시점에 채널별 통계 텍스트도 갱신해 overlay 에 사용한다.
        """

        for ch_id, w in enumerate(self.video_widgets):
            stats_text = self._get_channel_stats_snapshot(ch_id)
            w.update_stats_text(stats_text)
            w.update()

    # ----- GUI 슬롯 -----

    @QtCore.Slot(int, object, dict)
    def on_frame_ready(self, channel_id: int, frame_bgr: np.ndarray, meta: Dict[str, Any]) -> None:
        """frame_ready 시그널을 받아 VideoWidget 을 업데이트.

        - 여러 개의 frame_ready 가 밀려와도, 각 채널별로 항상 가장 최신 프레임만
          화면에 적용되도록 한다.
        """

        # 채널별 최신 프레임/메타를 갱신
        with self._latest_lock:
            self._latest_frames[channel_id] = frame_bgr
            self._latest_meta[channel_id] = meta

        # 현재 시점의 최신 프레임을 사용해 위젯을 업데이트
        if 0 <= channel_id < len(self.video_widgets):
            latest_frame = None
            latest_meta = None
            with self._latest_lock:
                latest_frame = self._latest_frames.get(channel_id)
                latest_meta = self._latest_meta.get(channel_id, {})

            if latest_frame is not None:
                self.video_widgets[channel_id].set_frame(latest_frame)

            # 성능 메트릭 및 채널별 통계 업데이트
            if latest_meta is not None:
                self._update_performance_metrics(latest_meta)
                self._update_channel_stats_on_frame(channel_id, latest_meta)

        # 전역 처리량(throughput) 요약 로그를 10초 간격으로 메인 스레드에서 한 번만 출력
        self._log_throughput_summary()

    # ----- 종료 처리 -----

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover
        """윈도우 닫힐 때 워커 및 캡처 스레드 정리."""
        print("[INFO] 종료 시작...")
        
        # 1. 모든 스레드에 종료 신호 전송
        self.stop_event.set()
        for t in self.capture_threads:
            if t:
                t.stop()
        
        # 2. 캡처 스레드 종료 대기
        for t in self.capture_threads:
            if t and t.is_alive():
                t.join(timeout=2.0)
        
        # 3. 워커 스레드에 종료 신호 전송 (None 전송)
        if hasattr(self, '_threads'):
            for _ in range(len(self._threads)):
                for q in [self.input_queue, self.infer_queue, self.draw_queue]:
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass
        
            # 4. 워커 스레드 종료 대기
            for t in self._threads:
                if t and t.is_alive():
                    t.join(timeout=1.0)
        
        # 5. 엔진 정리
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        
        print("[INFO] 종료 완료")
        event.accept()


# ===== 설정 파일 로딩 및 앱 실행 =====


def load_config(path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:  # pragma: no cover - 실행 엔트리
    base_dir = Path(__file__).resolve().parent
    default_cfg = base_dir / "config" / "yolo26_multich.yaml"

    if not default_cfg.exists():
        print(f"[ERROR] 설정 파일을 찾을 수 없습니다: {default_cfg}")
        sys.exit(1)

    config = load_config(str(default_cfg))

    # cv2.setNumThreads(1)

    app = QtWidgets.QApplication(sys.argv)

    # 다크 테마 적용
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

    # 스크롤 영역/체크박스 등 위젯들이 다크 테마에 잘 어울리도록 기본 스타일 유지

    win = MainWindow(config)
    win.resize(1280, 720)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
