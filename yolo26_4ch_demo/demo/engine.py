"""YOLO26 멀티 채널용 추론 엔진.

`yolo26_async.py` 와는 **완전히 독립적인** 구현으로,
멀티 채널 + 멀티 스레드 환경에서 사용하기 쉽게 설계한다.

- DX InferenceEngine 래핑
- preprocess / run_async / wait 제공
- pad/gain 등 상태는 **self 에 저장하지 않고**, frame 별 meta 로 관리
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine
from packaging import version

class YOLO26Engine:
    """멀티 채널에서 공용으로 사용할 YOLO26 엔진.

    - InferenceEngine 을 한 번만 생성해 여러 채널이 공유
    - 멀티 스레드 환경에서도 안전하도록, 프레임별 meta 로 좌표 변환 정보 관리
    """

    def __init__(self, model_path: str) -> None:
        # DX-RT 버전 체크 (필요 시 main 쪽에서 한 번만 호출하도록 바꿔도 됨)
        config = Configuration()
        if version.parse(config.get_version()) < version.parse("3.0.0"):
            raise RuntimeError(
                "DX-RT v3.0.0 이상이 필요합니다. DX-RT 를 업데이트 해주세요."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # # intra-op 1
        # config.set_enable(Configuration.ITEM.CUSTOM_INTRA_OP_THREADS, True)
        # config.set_attribute(Configuration.ITEM.CUSTOM_INTRA_OP_THREADS,
        #                     Configuration.ATTRIBUTE.CUSTOM_INTRA_OP_THREADS_NUM,
        #                     "1")

        # # inter-op 1
        # config.set_enable(Configuration.ITEM.CUSTOM_INTER_OP_THREADS, True)
        # config.set_attribute(Configuration.ITEM.CUSTOM_INTER_OP_THREADS,
        #                     Configuration.ATTRIBUTE.CUSTOM_INTER_OP_THREADS_NUM,
        #                     "1")

        # DXNN 모델 로드
        self.ie = InferenceEngine(model_path)

        if version.parse(self.ie.get_model_version()) < version.parse("7"):
            raise RuntimeError(
                ".dxnn 포맷 버전 7 이상이 필요합니다. DX-COM 을 업데이트 후 모델을 재컴파일 해주세요."
            )

        input_tensors_info = self.ie.get_input_tensors_info()
        # (N, H, W, C) 또는 (N, C, H, W) 를 가정하고, 예제와 동일하게 인덱스 사용
        self.input_height = input_tensors_info[0]["shape"][1]
        self.input_width = input_tensors_info[0]["shape"][2]

        # 탐지 관련 설정
        self.score_threshold = 0.4

        # COCO80 클래스 이름 및 색상 팔레트
        self.classes = [
                        "person",
                        "bicycle",
                        "car",
                        "motorcycle",
                        "airplane",
                        "bus",
                        "train",
                        "truck",
                        "boat",
                        "traffic light",
                        "fire hydrant",
                        "stop sign",
                        "parking meter",
                        "bench",
                        "bird",
                        "cat",
                        "dog",
                        "horse",
                        "sheep",
                        "cow",
                        "elephant",
                        "bear",
                        "zebra",
                        "giraffe",
                        "backpack",
                        "umbrella",
                        "handbag",
                        "tie",
                        "suitcase",
                        "frisbee",
                        "skis",
                        "snowboard",
                        "sports ball",
                        "kite",
                        "baseball bat",
                        "baseball glove",
                        "skateboard",
                        "surfboard",
                        "tennis racket",
                        "bottle",
                        "wine glass",
                        "cup",
                        "fork",
                        "knife",
                        "spoon",
                        "bowl",
                        "banana",
                        "apple",
                        "sandwich",
                        "orange",
                        "broccoli",
                        "carrot",
                        "hot dog",
                        "pizza",
                        "donut",
                        "cake",
                        "chair",
                        "couch",
                        "potted plant",
                        "bed",
                        "dining table",
                        "toilet",
                        "tv",
                        "laptop",
                        "mouse",
                        "remote",
                        "keyboard",
                        "cell phone",
                        "microwave",
                        "oven",
                        "toaster",
                        "sink",
                        "refrigerator",
                        "book",
                        "clock",
                        "vase",
                        "scissors",
                        "teddy bear",
                        "hair drier",
                        "toothbrush",
                    ]
        
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 현재 in-flight 비동기 요청 개수를 추적하기 위한 카운터
        self._inflight = 0

    # ===== 전처리 관련 =====

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """원본 비율을 유지하면서 모델 입력 크기에 맞게 letterbox.

        반환값:
          - padding 이 적용된 이미지
          - pad (top, left)
        """

        shape = img.shape[:2]  # (h, w)

        # 기존 예제와 동일한 비율 계산
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return img, (top, left)

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """단일 프레임 전처리.

        반환값:
          - input_tensor: run_async 에 넣을 입력 텐서 (모델 입력 크기)
          - meta: 후처리 시 필요한 정보 (원본 이미지 크기, pad, gain 등)
        """

        img_height, img_width = frame_bgr.shape[:2]

        # BGR → RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # letterbox 적용 (비율 유지 + padding)
        input_tensor, (pad_top, pad_left) = self.letterbox(
            img_rgb, (self.input_height, self.input_width)
        )

        # letterbox 에서 사용된 스케일 비율 계산
        # (letterbox 내부 계산과 동일한 방식으로 역산)
        r = min(self.input_height / img_height, self.input_width / img_width)

        meta: Dict[str, Any] = {
            "orig_shape": (img_height, img_width),  # (H, W)
            "pad_top": pad_top,
            "pad_left": pad_left,
            "scale": r,
        }
        return input_tensor, meta

    # ===== InferenceEngine 비동기 호출 =====

    def run_async(self, input_tensor: np.ndarray) -> int:
        """InferenceEngine.run_async 래핑.

        - 멀티 채널에서 공용 InferenceEngine 을 사용.
        """
        req_id = self.ie.run_async([input_tensor])

        # in-flight 카운터 증가
        self._inflight += 1

        # 디버깅용으로 너무 자주 찍히지 않도록, 간단한 샘플링만 수행
        # (예: 16개 단위로 증가할 때마다 한 번 출력)
        # print(f"[INF] InferenceEngine inflight requests: {self._inflight}")

        return req_id

    def wait(self, req_id: int) -> List[np.ndarray]:
        """InferenceEngine.wait 래핑."""
        outputs = self.ie.wait(req_id)

        # 요청 하나 완료되었으므로 in-flight 카운터 감소
        if self._inflight > 0:
            self._inflight -= 1

        return outputs

    # ===== 후처리 관련 =====

    @staticmethod
    def convert_to_original_coordinates(
        detections: np.ndarray, meta: Dict[str, Any]
    ) -> np.ndarray:
        """모델 입력 좌표계를 원본 이미지 좌표계로 변환.

        letterbox 전처리에서 사용된 padding, scale 정보를 meta 에서 읽어와
        원본 이미지 좌표계로 복원한다.
        """

        if detections is None or len(detections) == 0:
            return detections

        orig_h, orig_w = meta["orig_shape"]
        pad_top = meta["pad_top"]
        pad_left = meta["pad_left"]
        scale = meta["scale"]

        # letterbox: x, y 는 (pad_left, pad_top) 만큼 padding 된 뒤 scale 이 적용된 상태
        # 따라서 padding 을 제거하고 scale 을 되돌린다.
        detections[:, 0] = (detections[:, 0] - pad_left) / scale
        detections[:, 1] = (detections[:, 1] - pad_top) / scale
        detections[:, 2] = (detections[:, 2] - pad_left) / scale
        detections[:, 3] = (detections[:, 3] - pad_top) / scale

        # 원본 이미지 크기로 클리핑
        detections[:, 0] = np.clip(detections[:, 0], 0, orig_w - 1)
        detections[:, 1] = np.clip(detections[:, 1], 0, orig_h - 1)
        detections[:, 2] = np.clip(detections[:, 2], 0, orig_w - 1)
        detections[:, 3] = np.clip(detections[:, 3], 0, orig_h - 1)

        return detections

    # ===== 시각화 =====

    def draw_detections(
        self,
        img: np.ndarray,
        detections: np.ndarray,
        meta: Dict[str, Any]
    ) -> None:
        """프레임 위에 bounding box 시각화."""

        if detections is None or len(detections) == 0:
            return

        # self.score_threshold 적용
        detections = detections[detections[:, 4] >= self.score_threshold]

        # letterbox 전처리에 맞게 원본 좌표계로 변환
        detections = self.convert_to_original_coordinates(detections, meta)

        # bbox + 라벨 텍스트
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection

            color = self.color_palette[int(class_id)]

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            label = f"{self.classes[int(class_id)]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_x = int(x1)
            label_y = int(y1) - 10 if int(y1) - 10 > label_height else int(y1) + 10

            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
