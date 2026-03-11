"""YOLOv11 멀티 채널용 추론 엔진.

`yolov11_async.py` 와는 **완전히 독립적인** 구현으로,
멀티 채널 + 멀티 스레드 환경에서 사용하기 쉽게 설계한다.

- DX InferenceEngine 래핑
- preprocess / postprocess / run_async / wait 제공
- pad/gain 등 상태는 **self 에 저장하지 않고**, frame 별 meta 로 관리
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine
from dx_postprocess import YOLOv8SegPostProcess, overlay_segmentation
from packaging import version

class YOLOv11Engine:
    """멀티 채널에서 공용으로 사용할 YOLOv11 엔진.

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
        self.score_threshold = 0.5
        self.nms_threshold = 0.45

        self.postprocessor = YOLOv8SegPostProcess(
            self.input_width,
            self.input_height,
            self.score_threshold,
            self.nms_threshold,
            True
        )

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

        # 단순 resize 전처리: 종횡비는 유지하지 않고, 입력 크기에 맞게 스케일링
        # 좌표 변환 시에는 가로/세로 각각의 스케일 비율을 사용한다.
        gain_y = self.input_height / img_height
        gain_x = self.input_width / img_width

        # BGR → RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # letterbox 대신 단순 resize 만 적용
        input_tensor = cv2.resize(
            img_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )

        meta: Dict[str, Any] = {
            "orig_shape": (img_height, img_width),  # (H, W)
            # 종횡비를 유지하지 않고 resize 했으므로, 가로/세로 비율을 따로 저장한다.
            "gain_x": gain_x,
            "gain_y": gain_y,
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

    def _raw_postprocess(self, output_tensors: List[np.ndarray]) -> np.ndarray:
        """모델 출력 텐서에서 박스/클래스/스코어를 뽑는 부분.

        - 좌표는 아직 모델 입력 좌표계 기준(x1, y1, x2, y2)
        """

        outputs = np.transpose(np.squeeze(output_tensors[0]))

        cls_scores = outputs[:, 4:]
        cls_max_scores = np.max(cls_scores, axis=1)
        cls_ids = np.argmax(cls_scores, axis=1)

        # (center_x, center_y, width, height)
        boxes_cxcywh = outputs[:, :4]

        # (left, top, right, bottom)
        boxes_x1y1x2y2 = np.column_stack(
            [
                boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5,
                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5,
                boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] * 0.5,
                boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] * 0.5,
            ]
        )

        # NMS 를 위해 (left, top, width, height) 형태도 구성
        boxes_x1y1wh = np.column_stack(
            [
                boxes_x1y1x2y2[:, 0],
                boxes_x1y1x2y2[:, 1],
                boxes_x1y1x2y2[:, 2] - boxes_x1y1x2y2[:, 0],
                boxes_x1y1x2y2[:, 3] - boxes_x1y1x2y2[:, 1],
            ]
        )

        indices = cv2.dnn.NMSBoxes(
            boxes_x1y1wh.tolist(),
            cls_max_scores.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )

        if len(indices) > 0:
            keep = np.array(indices).reshape(-1)
            return np.column_stack(
                [boxes_x1y1x2y2[keep], cls_max_scores[keep], cls_ids[keep]]
            ).astype(np.float32)

        return np.empty((0, 6), dtype=np.float32)

    @staticmethod
    def convert_to_original_coordinates(
        detections: np.ndarray, meta: Dict[str, Any]
    ) -> np.ndarray:
        """모델 입력 좌표계를 원본 이미지 좌표계로 변환.

        - self 상태를 사용하지 않고, meta 에서 필요한 값을 읽어온다.
        """

        if len(detections) == 0:
            return detections

        orig_h, orig_w = meta["orig_shape"]
        gain_x = meta["gain_x"]
        gain_y = meta["gain_y"]

        # 단순 resize 전처리에서는 padding 이 없고,
        # 가로/세로 스케일 비율이 서로 다를 수 있으므로 각각을 사용해 역변환한다.
        detections[:, 0] = np.clip(
            detections[:, 0] / gain_x, 0, orig_w - 1
        )
        detections[:, 1] = np.clip(
            detections[:, 1] / gain_y, 0, orig_h - 1
        )
        detections[:, 2] = np.clip(
            detections[:, 2] / gain_x, 0, orig_w - 1
        )
        detections[:, 3] = np.clip(
            detections[:, 3] / gain_y, 0, orig_h - 1
        )

        return detections

    def _transform_single_mask_to_original(
        self,
        mask: np.ndarray,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """단일 mask 를 모델 입력 해상도 → 시각화용 해상도로 변환.

        - mask: (H_in, W_in) uint8 또는 float
        - meta: preprocess 에서 넘어온 orig_shape, gain_x, gain_y 포함

        CES 데모 품질을 유지하면서 성능을 확보하기 위해,
        원본 해상도 전체 대신 약 0.5x 수준의 중간 해상도로 변환한다.
        """

        orig_h, orig_w = meta["orig_shape"]  # (H, W)

        # mask 를 0~255 범위 uint8 로 정규화
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0.0, 1.0)
            mask = (mask * 255.0).astype(np.uint8)

        # resize-only 전처리에서는 pad 가 없으므로, mask 전체를 사용한다.
        in_h, in_w = mask.shape[:2]

        # 시각화용 중간 해상도 (대략 0.5x 수준으로 조정)
        # 너무 낮추면 계단 현상이 심해지므로, min 으로 하한도 함께 고려.
        target_w = max(640, orig_w // 2)
        target_h = max(360, orig_h // 2)

        # 중간 해상도로 리사이즈 후 이진화
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        _, mask_binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return mask_binary

    def _transform_masks_to_original(
        self,
        masks_input_space: np.ndarray,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """여러 개의 mask 를 원본 이미지 해상도로 변환.

        - masks_input_space: (N, H_in, W_in)
        - 반환: (N, orig_h, orig_w) uint8
        """

        if masks_input_space is None or len(masks_input_space) == 0:
            orig_h, orig_w = meta["orig_shape"]
            return np.empty((0, orig_h, orig_w), dtype=np.uint8)

        masks_orig = []
        for m in masks_input_space:
            m_orig = self._transform_single_mask_to_original(m, meta)
            masks_orig.append(m_orig)

        return np.stack(masks_orig, axis=0)

    def postprocess(
        self, output_tensors: List[np.ndarray], meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """네트워크 출력 텐서 후처리.

        - C++ YOLOv8SegPostProcess 로 bbox + mask 생성
        - bbox/mask 를 모두 원본 이미지 좌표계/해상도로 변환
        """

        # C++ postprocess 호출: (detections, masks) 튜플 반환
        dets_input_space, masks_input_space = self.postprocessor.postprocess(
            output_tensors
        )

        # bbox 를 원본 이미지 좌표계로 변환
        dets_orig = self.convert_to_original_coordinates(
            dets_input_space.copy(),  # in-place 수정이므로 copy 사용
            meta,
        )

        # mask 를 원본 해상도로 변환
        masks_orig = self._transform_masks_to_original(
            masks_input_space,
            meta,
        )

        return dets_orig, masks_orig

    # ===== 시각화 =====

    def apply_segmentation_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        detections: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """세그멘테이션 마스크를 반투명 오버레이로 적용.

        - image: BGR 원본 이미지 (H, W, 3) uint8
        - masks: (N, H, W) uint8, 시각화 해상도
        - detections: (N, 6) [x1,y1,x2,y2,score,class_id]

        실제 연산은 C++ 확장 모듈(dx_postprocess.overlay_segmentation)에서 수행한다.
        """

        if masks is None or len(masks) == 0:
            return image

        img_h, img_w = image.shape[:2]

        # C++ 쪽에서 H,W가 다르면 예외를 던지므로, 여기서 한 번 맞춰준다.
        if masks.shape[1] != img_h or masks.shape[2] != img_w:
            resized_masks = np.zeros((masks.shape[0], img_h, img_w), dtype=np.uint8)
            for i in range(masks.shape[0]):
                resized_masks[i] = cv2.resize(
                    masks[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )
            masks_for_overlay = resized_masks
        else:
            masks_for_overlay = masks

        # detections 는 float32, palette 는 uint8 로 맞춰서 넘긴다.
        dets = detections.astype(np.float32, copy=False)
        palette = self.color_palette.astype(np.uint8, copy=False)

        return overlay_segmentation(image, masks_for_overlay, dets, palette, float(alpha))

    def draw_detections(
        self,
        img: np.ndarray,
        detections: np.ndarray,
        masks: np.ndarray | None = None,
    ) -> None:
        """프레임 위에 bounding box + segmentation mask 시각화."""

        # 1) 마스크 오버레이
        if masks is not None and len(masks) > 0:
            img_overlay = self.apply_segmentation_overlay(img, masks, detections)
        else:
            img_overlay = img

        # 2) bbox + 라벨 텍스트
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection

            color = self.color_palette[int(class_id)]

            cv2.rectangle(
                img_overlay,
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
                img_overlay,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                img_overlay,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 최종 결과를 원본 img 에 반영 (in-place)
        if img_overlay is not img:
            img[:, :, :] = img_overlay
