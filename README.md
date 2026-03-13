# DX-DEMO

A collection of demo applications for DEEPX NPU inference.

## Demos

| Demo | Model | Description |
|------|-------|-------------|
| [dx-clip-demo](https://github.com/DEEPX-AI/dx-clip-demo) | CLIP | Real-time text-video similarity matching powered by CLIP on DeepX NPU |
| [yolo11seg_4ch_demo](yolo11seg_4ch_demo/README.md) | YOLOv11 Segmentation | Real-time instance segmentation with mask overlay across up to 4 input channels |
| [yolo26_4ch_demo](yolo26_4ch_demo/README.md) | YOLO26 | Real-time object detection with per-class BBOX toggle panel across up to 4 input channels |

## Screenshots

### DX-CLIP Demo

![DX-CLIP Demo Screenshot](img/dx-clip-demo_screenshot.png)

### YOLOv11 Segmentation 4-Channel Demo

![YOLOv11 Segmentation 4-Channel Demo Screenshot](yolo11seg_4ch_demo/img/yolov11seg_4ch_demo_screenshot.png)

### YOLO26 4-Channel Demo

![YOLO26 4-Channel Demo Screenshot](yolo26_4ch_demo/img/yolo26_4ch_demo_screenshot.png)

## Prerequisites

All demos require **DX-RT** (DeepX Runtime) to be built and installed before use.

```python
# Verify DX-RT is available
import dx_engine
```

## Demo Details

- **[dx-clip-demo](https://github.com/DEEPX-AI/dx_clip_demo)** — Real-time text-video similarity matching using the CLIP model accelerated on DeepX NPU. Supports up to 16 video channels, camera input, and configurable GUI options.
- **[yolo11seg_4ch_demo](yolo11seg_4ch_demo/README.md)** — Uses a C++ Python binding (`dx_postprocess`) to accelerate pixel-level mask overlay operations for real-time multi-channel segmentation.
- **[yolo26_4ch_demo](yolo26_4ch_demo/README.md)** — Features a class list panel in the Qt GUI with per-class checkboxes to toggle BBOX display individually. Uses the YOLO26 detection model.