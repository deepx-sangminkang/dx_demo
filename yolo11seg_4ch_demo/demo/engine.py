"""Inference engine for YOLOv11 multi-channel execution.

This implementation is **fully independent** of `yolov11_async.py`
and is designed for convenient use in multi-channel, multi-threaded environments.

- Wraps DX InferenceEngine
- Provides preprocess / postprocess / run_async / wait
- Keeps state such as pad/gain out of `self` and manages it per frame in meta
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
    """YOLOv11 engine shared across multiple channels.

    - Builds InferenceEngine once and shares it across channels
    - Stores coordinate-transform metadata per frame for thread safety
    """

    def __init__(self, model_path: str) -> None:
        # Check the DX-RT version here, or move it to main if it should only run once.
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

        # Load the DXNN model
        self.ie = InferenceEngine(model_path)

        if version.parse(self.ie.get_model_version()) < version.parse("7"):
            raise RuntimeError(
                ".dxnn format version 7 or later is required. Please update DX-COM and recompile the model."
            )

        input_tensors_info = self.ie.get_input_tensors_info()
        # Assume (N, H, W, C) or (N, C, H, W), using the same indexing as the example.
        self.input_height = input_tensors_info[0]["shape"][1]
        self.input_width = input_tensors_info[0]["shape"][2]

        # Detection settings
        self.score_threshold = 0.5
        self.nms_threshold = 0.45

        self.postprocessor = YOLOv8SegPostProcess(
            self.input_width,
            self.input_height,
            self.score_threshold,
            self.nms_threshold,
            True
        )

        # COCO80 class names and colour palette
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

        # Track the number of async requests currently in flight
        self._inflight = 0

    # ===== Preprocessing =====

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Apply letterbox to fit the model input while preserving aspect ratio.

        Returns:
          - image with padding applied
          - pad (top, left)
        """

        shape = img.shape[:2]  # (h, w)

        # Use the same ratio calculation as the existing example.
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
        """Preprocess a single frame.

        Returns:
          - input_tensor: input tensor passed to run_async (model input size)
          - meta: information needed for postprocess (original image size, pad, gain, etc.)
        """

        img_height, img_width = frame_bgr.shape[:2]

        # Resize-only preprocess: ignore aspect ratio and scale directly to input size.
        # Use separate horizontal and vertical scale factors during coordinate conversion.
        gain_y = self.input_height / img_height
        gain_x = self.input_width / img_width

        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply a simple resize instead of letterbox.
        input_tensor = cv2.resize(
            img_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        meta: Dict[str, Any] = {
            "orig_shape": (img_height, img_width),  # (H, W)
            # Aspect ratio is not preserved, so store separate horizontal and vertical ratios.
            "gain_x": gain_x,
            "gain_y": gain_y,
        }
        return input_tensor, meta

    # ===== InferenceEngine async calls =====

    def run_async(self, input_tensor: np.ndarray) -> int:
        """Wrapper around InferenceEngine.run_async.

        - Uses the shared InferenceEngine across multiple channels.
        """
        req_id = self.ie.run_async([input_tensor])

        # Increment the in-flight counter
        self._inflight += 1

        # Use simple sampling for debugging to avoid logging too frequently.
        # (For example, print once whenever the count increases by 16.)
        # print(f"[INF] InferenceEngine inflight requests: {self._inflight}")

        return req_id

    def wait(self, req_id: int) -> List[np.ndarray]:
        """Wrapper around InferenceEngine.wait."""
        outputs = self.ie.wait(req_id)

        # One request completed, so decrement the in-flight counter
        if self._inflight > 0:
            self._inflight -= 1

        return outputs

    # ===== Postprocessing =====

    def _raw_postprocess(self, output_tensors: List[np.ndarray]) -> np.ndarray:
        """Extract boxes, classes, and scores from model output tensors.

        - Coordinates are still in the model-input coordinate space (x1, y1, x2, y2).
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

        # Also build (left, top, width, height) boxes for NMS
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
        """Convert model-input coordinates to original-image coordinates.

        - Reads the required values from meta instead of using engine state.
        """

        if len(detections) == 0:
            return detections

        orig_h, orig_w = meta["orig_shape"]
        gain_x = meta["gain_x"]
        gain_y = meta["gain_y"]

        # Resize-only preprocessing has no padding,
        # and horizontal/vertical scale factors may differ, so reverse them separately.
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
        """Convert a single mask from model-input resolution to visualisation resolution.

        - mask: (H_in, W_in) uint8 or float
        - meta: includes orig_shape, gain_x, and gain_y from preprocess

        To balance quality and performance for the CES demo,
        convert to an intermediate resolution of roughly 0.5x instead of the full original size.
        """

        orig_h, orig_w = meta["orig_shape"]  # (H, W)

        # Normalise mask to uint8 in the 0..255 range
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0.0, 1.0)
            mask = (mask * 255.0).astype(np.uint8)

        # Resize-only preprocessing has no padding, so use the full mask.
        in_h, in_w = mask.shape[:2]

        # Intermediate resolution for visualisation (roughly 0.5x).
        # Keep a lower bound so aliasing does not become too severe.
        target_w = max(640, orig_w // 2)
        target_h = max(360, orig_h // 2)

        # Resize to the intermediate resolution, then binarise
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        _, mask_binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return mask_binary

    def _transform_masks_to_original(
        self,
        masks_input_space: np.ndarray,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """Convert multiple masks to original-image resolution.

        - masks_input_space: (N, H_in, W_in)
        - returns: (N, orig_h, orig_w) uint8
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
        """Postprocess network output tensors.

        - Uses C++ YOLOv8SegPostProcess to produce bbox + mask results
        - Converts both bbox and mask data to original-image coordinates/resolution
        """

        # Call C++ postprocess: returns a (detections, masks) tuple
        dets_input_space, masks_input_space = self.postprocessor.postprocess(
            output_tensors
        )

        # Convert bbox data to original-image coordinates
        dets_orig = self.convert_to_original_coordinates(
            dets_input_space.copy(),  # Use copy because the conversion modifies the array in place
            meta,
        )

        # Convert masks to original resolution
        masks_orig = self._transform_masks_to_original(
            masks_input_space,
            meta,
        )

        return dets_orig, masks_orig

    # ===== Visualisation =====

    def apply_segmentation_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        detections: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Apply segmentation masks as a semi-transparent overlay.

        - image: original BGR image (H, W, 3) uint8
        - masks: (N, H, W) uint8, visualisation resolution
        - detections: (N, 6) [x1,y1,x2,y2,score,class_id]

        The actual pixel operation runs in the C++ extension module
        (`dx_postprocess.overlay_segmentation`).
        """

        if masks is None or len(masks) == 0:
            return image

        img_h, img_w = image.shape[:2]

        # The C++ side raises if H/W differ, so align them here first.
        if masks.shape[1] != img_h or masks.shape[2] != img_w:
            resized_masks = np.zeros((masks.shape[0], img_h, img_w), dtype=np.uint8)
            for i in range(masks.shape[0]):
                resized_masks[i] = cv2.resize(
                    masks[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )
            masks_for_overlay = resized_masks
        else:
            masks_for_overlay = masks

        # Pass detections as float32 and palette as uint8.
        dets = detections.astype(np.float32, copy=False)
        palette = self.color_palette.astype(np.uint8, copy=False)

        return overlay_segmentation(image, masks_for_overlay, dets, palette, float(alpha))

    def draw_detections(
        self,
        img: np.ndarray,
        detections: np.ndarray,
        masks: np.ndarray | None = None,
    ) -> None:
        """Draw bounding boxes and segmentation masks on a frame."""

        # 1) Mask overlay
        if masks is not None and len(masks) > 0:
            img_overlay = self.apply_segmentation_overlay(img, masks, detections)
        else:
            img_overlay = img

        # 2) Bounding boxes + label text
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

        # Apply the final result back to the original img (in-place)
        if img_overlay is not img:
            img[:, :, :] = img_overlay
