"""Display-side metadata for the dx_stream (native) backend.

Inference runs entirely inside the native ``dxinfer`` GStreamer element, so the
Python side only needs class names, a colour palette, and the model input size
to draw overlays. No NPU model is loaded here (and no OpenCV dependency).
"""

from __future__ import annotations

import numpy as np

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
