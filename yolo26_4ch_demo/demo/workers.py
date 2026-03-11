"""멀티 채널 YOLO26 파이프라인 워커/스레드 스켈레톤.

C 방식 아키텍처 (채널별 캡처 + 전역 pre/infer/draw 워커)를 구현하기 위한 뼈대 코드.
- 실제 로직은 간결하게 유지하고, 역할이 잘 드러나도록 한글 주석 위주로 작성.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections import defaultdict

import cv2
import numpy as np

from .engine import YOLO26Engine


# ===== 단계별 드롭 카운트 (단순/직관적 버전) =====

# queue_drop_counts[stage][channel_id] = count
queue_drop_counts: Dict[str, Dict[int, int]] = {
    "input": defaultdict(int),      # input_queue (capture 단계)
    "infer": defaultdict(int),      # infer_queue (preprocess 단계)
    "draw": defaultdict(int),       # draw_queue (draw 단계)
}
queue_drop_lock = threading.Lock()


# ===== 단계별 처리량(throughput) 통계 =====

throughput_stats: Dict[str, Dict[str, Any]] = {
    "read": {"first_ts": None, "last_ts": None, "count": 0},
    "pre": {"first_ts": None, "last_ts": None, "count": 0},
    "inf": {"first_ts": None, "last_ts": None, "count": 0},
    "draw": {"first_ts": None, "last_ts": None, "count": 0},
}
throughput_lock = threading.Lock()


def record_throughput(stage: str, ts: float) -> None:
    """단계별 처리량 통계를 업데이트.

    - 첫 처리 시점(first_ts), 마지막 처리 시점(last_ts), 처리된 프레임 수(count)를 기록한다.
    """

    with throughput_lock:
        s = throughput_stats.get(stage)
        if s is None:
            return
        if s["first_ts"] is None:
            s["first_ts"] = ts
        s["last_ts"] = ts
        s["count"] += 1


def get_fps(stage: str) -> float:
    """저장된 통계로부터 해당 단계의 FPS 를 계산한다."""

    with throughput_lock:
        s = throughput_stats.get(stage)
        if not s:
            return 0.0
        first_ts = s["first_ts"]
        last_ts = s["last_ts"]
        count = s["count"]

    if first_ts is None or last_ts is None or last_ts <= first_ts or count <= 0:
        return 0.0

    return count / (last_ts - first_ts)


# ===== 공용 큐에 넣을 데이터 구조 =====


@dataclass
class CaptureItem:
    """캡처 스레드 → preprocess_worker 로 넘어가는 데이터.

    channel_id: 어느 채널에서 온 프레임인지 구분 용도
    frame_bgr: 원본 BGR 프레임 (시각화용으로 보관)
    meta: 타임스탬프 등 부가 정보
    """

    channel_id: int
    frame_bgr: np.ndarray
    meta: Dict[str, Any]


@dataclass
class InferItem:
    """preprocess_worker → wait_worker 로 넘어가는 데이터."""

    channel_id: int
    frame_bgr: np.ndarray
    input_tensor: np.ndarray
    req_id: int
    meta: Dict[str, Any]


@dataclass
class OutputItem:
    """wait_worker → draw_worker 로 넘어가는 데이터."""

    channel_id: int
    frame_bgr: np.ndarray
    output_tensors: Any
    meta: Dict[str, Any]


# ===== 채널별 캡처 스레드 =====


class CaptureThread(threading.Thread):
    """각 채널별로 하나씩 생성되는 캡처 스레드.

    - USB Cam / 비디오 파일 / RTSP 에서 프레임을 읽어 공용 input_queue 에 넣는다.
    - DX 추론과 GUI 업데이트는 다른 워커/스레드에서 담당.
    """

    def __init__(
        self,
        channel_id: int,
        source: Any,
        input_queue: "queue.Queue[CaptureItem]",
        max_fps: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(daemon=True, name=name or f"CaptureThread-{channel_id}")
        self.channel_id = channel_id
        self.source = source
        self.input_queue = input_queue
        self.max_fps = max_fps
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """외부에서 스레드 종료 요청."""

        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - 실제 런타임 전용
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] 채널 {self.channel_id}: 입력 소스를 열 수 없습니다 - {self.source}")
            return

        print(f"[INFO] 채널 {self.channel_id}: 캡처 시작 - {self.source}")

        min_interval = 1.0 / self.max_fps if self.max_fps and self.max_fps > 0 else 0.0

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()
                if not ok:
                    # 비디오 파일의 경우 끝까지 도달하면 다시 처음으로 되감아서 무한 반복 재생.
                    # RTSP/카메라 등에서는 EOF 개념이 없거나 오류일 수 있지만,
                    # 파일 경로(str) 입력이 일반적인 데모 시나리오이므로 우선 파일 기준으로 처리.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        print(
                            f"[INFO] 채널 {self.channel_id}: 프레임을 더 이상 읽을 수 없습니다 (EOF 또는 오류)"
                        )
                        break

                meta: Dict[str, Any] = {
                    "t_read": time.perf_counter() - t0,
                    "ts": time.time(),
                }

                item = CaptureItem(
                    channel_id=self.channel_id,
                    frame_bgr=frame_bgr,
                    meta=meta,
                )

                # 큐가 가득 찬 경우, 가장 오래된 프레임을 버리고 최신 프레임을 넣는 방식으로
                # "실시간성"을 유지한다. 이렇게 하면 입력 FPS > 처리 FPS 인 상황에서도
                # 뚝뚝 끊기는 느낌이 줄어들고, 항상 최신에 가까운 프레임이 화면에 표시된다.
                try:
                    self.input_queue.put(item, timeout=0.001)
                except queue.Full:
                    try:
                        # 하나 버리고 (오래된 프레임 제거)
                        dropped_item = self.input_queue.get_nowait()
                        # 캡처 단계에서 드롭된 프레임 카운트 증가
                        with queue_drop_lock:
                            queue_drop_counts["input"][self.channel_id] += 1
                    except queue.Empty:
                        dropped_item = None

                    try:
                        # 다시 최신 프레임을 넣는다.
                        self.input_queue.put_nowait(item)
                    except queue.Full:
                        # 극단적인 상황에서는 조용히 스킵 (로그 남기지 않음)
                        pass

                # read 단계 처리량 기록 (성공적으로 큐에 넣은 경우에 한해)
                record_throughput("read", time.time())

                # FPS 제한이 설정된 경우 간단한 sleep 적용
                if min_interval > 0.0:
                    elapsed = time.perf_counter() - t0
                    remain = min_interval - elapsed
                    if remain > 0:
                        time.sleep(remain)
        finally:
            cap.release()
            print(f"[INFO] 채널 {self.channel_id}: 캡처 종료")


# ===== 전역 워커 스레드 함수 =====


def preprocess_worker(
    engine: YOLO26Engine,
    input_queue: "queue.Queue[CaptureItem]",
    infer_queue: "queue.Queue[InferItem]",
    stop_event: threading.Event,
) -> None:
    """전역 전처리 + run_async 워커.

    - 여러 채널에서 들어오는 프레임을 하나의 큐에서 받아서 처리
    - preprocess 후 run_async 를 호출하고, req_id 와 함께 infer_queue 로 넘김
    """

    while not stop_event.is_set():  # pragma: no cover - 런타임 전용
        try:
            item = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item.frame_bgr is None:
            continue

        t0 = time.perf_counter()
        input_tensor, meta_pre = engine.preprocess(item.frame_bgr)
        item.meta.update(meta_pre)
        item.meta["t_preprocess"] = time.perf_counter() - t0

        # run_async 까지 포함해 전처리 단계 완료로 본다.
        req_id = engine.run_async(input_tensor)
        infer_item = InferItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            input_tensor=input_tensor,
            req_id=req_id,
            meta=item.meta,
        )

        # infer_queue 가 가득 찬 경우, 가장 오래된 항목을 하나 버리고 최신 항목을 넣는다.
        try:
            infer_queue.put(infer_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = infer_queue.get_nowait()
                # preprocess 단계(infer_queue)에서 드롭된 프레임 카운트 증가
                with queue_drop_lock:
                    queue_drop_counts["infer"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                infer_queue.put_nowait(infer_item)
            except queue.Full:
                # 극단적인 상황에서는 조용히 스킵 (로그 남기지 않음)
                pass

        # preprocess 단계 처리량 기록 (전처리 + run_async 완료 시점)
        record_throughput("pre", time.time())


def wait_worker(
    engine: YOLO26Engine,
    infer_queue: "queue.Queue[InferItem]",
    draw_queue: "queue.Queue[OutputItem]",
    stop_event: threading.Event,
) -> None:
    """전역 wait/inference 워커.

    - run_async 로 보낸 req_id 를 기다렸다가 output_tensors 를 받아 output_queue 로 넘김
    """

    while not stop_event.is_set():  # pragma: no cover - 런타임 전용
        try:
            item = infer_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = time.perf_counter()
        output_tensors = engine.wait(item.req_id)
        item.meta["t_inference"] = time.perf_counter() - t0

        out_item = OutputItem(
            channel_id=item.channel_id,
            frame_bgr=item.frame_bgr,
            output_tensors=output_tensors,
            meta=item.meta,
        )

        # draw_queue 도 가득 찬 경우, 가장 오래된 항목을 하나 버리고 최신 항목을 넣는다.
        try:
            draw_queue.put(out_item, timeout=0.001)
        except queue.Full:
            try:
                dropped_item = draw_queue.get_nowait()
                # draw 단계(draw_queue)에서 드롭된 프레임 카운트 증가
                with queue_drop_lock:
                    queue_drop_counts["draw"][dropped_item.channel_id] += 1
            except queue.Empty:
                dropped_item = None

            try:
                draw_queue.put_nowait(out_item)
            except queue.Full:
                # 극단적인 상황에서는 조용히 스킵 (로그 남기지 않음)
                pass

        # inference 단계 처리량 기록 (wait 완료 후 draw_queue 에 넣은 시점)
        record_throughput("inf", time.time())


def draw_worker(
    engine: YOLO26Engine,
    draw_queue: "queue.Queue[OutputItem]",
    get_selected_classes,
    on_frame_ready,
    stop_event: threading.Event,
) -> None:
    """전역 draw + GUI 전달 워커.

    - draw_queue 에서 프레임 + meta(detections) 를 받아
      draw_detections 를 호출하고 on_frame_ready 로 전달한다.
    """

    last_log_ts = time.time()

    while not stop_event.is_set():  # pragma: no cover - 런타임 전용
        try:
            item = draw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detections = np.squeeze(item.output_tensors)

        # 현재 선택된 클래스 집합을 읽어와 필터링
        selected_classes = get_selected_classes()

        if selected_classes is None:
            filtered_detections = detections
        elif len(selected_classes) == 0:
            # 아무 클래스도 선택되지 않은 경우 → 박스를 그리지 않음
            filtered_detections = np.empty((0, 6), dtype=detections.dtype)
        else:
            cls_mask = np.isin(detections[:, 5].astype(int), list(selected_classes))
            filtered_detections = detections[cls_mask]

        t_draw0 = time.perf_counter()
        engine.draw_detections(item.frame_bgr, filtered_detections, item.meta)
        t_draw1 = time.perf_counter()

        item.meta["t_draw"] = t_draw1 - t_draw0

        on_frame_ready(item.channel_id, item.frame_bgr, item.meta)

        # draw 단계 처리량 기록 (프레임 시각화 및 GUI 전달까지 완료한 시점)
        now = time.time()
        record_throughput("draw", now)
