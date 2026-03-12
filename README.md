# DEEPX DX-DEMO

A collection of demo applications for DEEPX NPU inference.

## Demos

| Demo | Model | Description |
|------|-------|-------------|
| [yolo11seg_4ch_demo](yolo11seg_4ch_demo/README.md) | YOLOv11 Segmentation | Real-time instance segmentation with mask overlay across up to 4 input channels |
| [yolo26_4ch_demo](yolo26_4ch_demo/README.md) | YOLO26 | Real-time object detection with per-class BBOX toggle panel across up to 4 input channels |

## Screenshots

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

## Quick Start

### 1. Navigate to the demo directory

```bash
# YOLOv11 Segmentation demo
cd yolo11seg_4ch_demo

# YOLO26 detection demo
cd yolo26_4ch_demo
```

### 2. Configure input sources

Edit the YAML config file inside `demo/config/` to set your model path and input channels (video file, RTSP stream, or camera).

### 3. Install and run

```bash
./run_demo.sh
```

`run_demo.sh` runs `install.sh` automatically on first launch, then starts the demo.

## Input Types

All demos support the following input types per channel:

| Type | Source Value | Example |
|------|-------------|---------|
| `video` | File path | `assets/videos/example.mov` |
| `rtsp` | Stream URL | `rtsp://192.168.1.100:8554/stream` |
| `camera` | Device index | `0` |

## Demo Details

- **[yolo11seg_4ch_demo](yolo11seg_4ch_demo/README.md)** — Uses a C++ Python binding (`dx_postprocess`) to accelerate pixel-level mask overlay operations for real-time multi-channel segmentation.
- **[yolo26_4ch_demo](yolo26_4ch_demo/README.md)** — Features a class list panel in the Qt GUI with per-class checkboxes to toggle BBOX display individually. Uses the YOLO26 detection model.