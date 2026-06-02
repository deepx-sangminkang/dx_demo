"""Inference engine for YOLO26 multi-channel use.

An implementation **completely independent** of `yolo26_async.py`,
designed for ease of use in a multi-channel + multi-thread environment.

- Wraps DX InferenceEngine
- Provides preprocess / run_async / wait
- State such as pad/gain is **not stored on self** but managed as per-frame meta
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from packaging import version

COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class NativeDisplayMeta:
    """Lightweight overlay metadata for the dxstream backend.

    Supplies class names, colour palette, and model input size **without**
    loading the NPU model in Python (inference runs in the native dxinfer
    element), so the Python engine doesn't contend for the NPU.
    """

    def __init__(self, input_width: int = 640, input_height: int = 640) -> None:
        self.classes = list(COCO80_CLASSES)
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3)
        )
        self.input_width = int(input_width)
        self.input_height = int(input_height)


class YOLO26Engine:
    """YOLO26 engine shared across multiple channels.

    - InferenceEngine is instantiated once and shared by all channels
    - Coordinate-transform info is managed per-frame in meta for thread safety
    """

    def __init__(self, model_path: str) -> None:
        # dx_engine (DX-RT Python module) is only needed for the legacy backend.
        # Import it lazily so the dxstream backend (which uses NativeDisplayMeta
        # and runs inference in the native dxinfer element) does not require it.
        from dx_engine import Configuration, InferenceEngine

        # DX-RT version check (can be moved to main to run once if needed)
        config = Configuration()
        if version.parse(config.get_version()) < version.parse("3.0.0"):
            raise RuntimeError(
                "DX-RT v3.0.0 or later is required. Please update DX-RT."
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

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

        # Load DXNN model
        self.ie = InferenceEngine(model_path)

        if version.parse(self.ie.get_model_version()) < version.parse("7"):
            raise RuntimeError(
                ".dxnn format version 7 or later is required. Please update DX-COM and recompile the model."
            )

        input_tensors_info = self.ie.get_input_tensors_info()
        # Assume (N, H, W, C) or (N, C, H, W), using the same indexing as in the example
        self.input_height = input_tensors_info[0]["shape"][1]
        self.input_width = input_tensors_info[0]["shape"][2]

        # Detection settings
        self.score_threshold = 0.4

        # COCO80 class names and colour palette
        self.classes = list(COCO80_CLASSES)

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Counter for tracking the number of currently in-flight async requests
        self._inflight = 0

    # ===== Preprocessing =====

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Apply letterbox to fit the model input size while preserving aspect ratio.

        Returns:
          - padded image
          - pad (top, left)
        """

        shape = img.shape[:2]  # (h, w)

        # Scale ratio calculation identical to the existing example
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

    def preprocess(
        self, frame: np.ndarray, color_format: str = "bgr"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess a single frame.

        Args:
          frame: captured frame. Assumed BGR unless ``color_format`` is ``"rgb"``
            (e.g. when the RGA dxconvert HW decode path already produced RGB),
            in which case the BGR->RGB conversion is skipped.

        Returns:
          - input_tensor: input tensor to pass to run_async (model input size)
          - meta: information needed for postprocessing (original image size, pad, gain, etc.)
        """

        img_height, img_width = frame.shape[:2]

        # Convert to RGB only when the source is not already RGB.
        if color_format == "rgb":
            img_rgb = frame
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply letterbox (maintain aspect ratio + padding)
        input_tensor, (pad_top, pad_left) = self.letterbox(
            img_rgb, (self.input_height, self.input_width)
        )

        # Compute the scale ratio used inside letterbox
        # (back-calculated the same way as the internal letterbox computation)
        r = min(self.input_height / img_height, self.input_width / img_width)

        meta: Dict[str, Any] = {
            "orig_shape": (img_height, img_width),  # (H, W)
            "pad_top": pad_top,
            "pad_left": pad_left,
            "scale": r,
        }
        return input_tensor, meta

    # ===== InferenceEngine async calls =====

    def run_async(self, input_tensor: np.ndarray) -> int:
        """Wrapper for InferenceEngine.run_async.

        - Uses the shared InferenceEngine across multiple channels.
        """
        req_id = self.ie.run_async([input_tensor])

        # Increment in-flight counter
        self._inflight += 1

        # Use simple sampling to avoid printing too frequently during debugging
        # (e.g. print once every time the count increases by 16)
        # print(f"[INF] InferenceEngine inflight requests: {self._inflight}")

        return req_id

    def wait(self, req_id: int) -> List[np.ndarray]:
        """Wrapper for InferenceEngine.wait."""
        outputs = self.ie.wait(req_id)

        # One request completed, so decrement in-flight counter
        if self._inflight > 0:
            self._inflight -= 1

        return outputs

    # ===== Postprocessing =====

    @staticmethod
    def convert_to_original_coordinates(
        detections: np.ndarray, meta: Dict[str, Any]
    ) -> np.ndarray:
        """Convert model-input coordinates to original image coordinates.

        Reads padding and scale information used during letterbox preprocessing from meta
        and restores the original image coordinate system.
        """

        if detections is None or len(detections) == 0:
            return detections

        orig_h, orig_w = meta["orig_shape"]
        pad_top = meta["pad_top"]
        pad_left = meta["pad_left"]
        scale = meta["scale"]

        # letterbox: x, y are in a state where padding (pad_left, pad_top) was applied then scaled
        # so remove the padding and reverse the scale.
        detections[:, 0] = (detections[:, 0] - pad_left) / scale
        detections[:, 1] = (detections[:, 1] - pad_top) / scale
        detections[:, 2] = (detections[:, 2] - pad_left) / scale
        detections[:, 3] = (detections[:, 3] - pad_top) / scale

        # Clip to original image dimensions
        detections[:, 0] = np.clip(detections[:, 0], 0, orig_w - 1)
        detections[:, 1] = np.clip(detections[:, 1], 0, orig_h - 1)
        detections[:, 2] = np.clip(detections[:, 2], 0, orig_w - 1)
        detections[:, 3] = np.clip(detections[:, 3], 0, orig_h - 1)

        return detections

    # ===== Visualisation =====

    def finalize_detections(
        self,
        output_tensors: Any,
        meta: Dict[str, Any],
        selected_classes: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Produce filtered detections in original-image coordinates.

        Applies class filtering, score thresholding, and the letterbox-inverse
        coordinate conversion so the GUI can overlay boxes without modifying the
        displayed frame (display-inference decoupling).
        """

        detections = np.squeeze(output_tensors)
        if detections is None or detections.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)

        if selected_classes is not None:
            if len(selected_classes) == 0:
                return np.empty((0, 6), dtype=detections.dtype)
            cls_mask = np.isin(detections[:, 5].astype(int), list(selected_classes))
            detections = detections[cls_mask]

        if len(detections) > 0:
            detections = detections[detections[:, 4] >= self.score_threshold]

        return self.convert_to_original_coordinates(detections, meta)

    def draw_detections(
        self,
        img: np.ndarray,
        detections: np.ndarray,
        meta: Dict[str, Any]
    ) -> None:
        """Visualise bounding boxes on a frame."""

        if detections is None or len(detections) == 0:
            return

        # Apply self.score_threshold
        detections = detections[detections[:, 4] >= self.score_threshold]

        # Convert to original coordinate system to match letterbox preprocessing
        detections = self.convert_to_original_coordinates(detections, meta)

        # bbox + label text
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
