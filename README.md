# dx_demo

A collection of multi-channel Qt demo applications for DeepX NPU inference.

## Demos

| Demo | Model | Description |
|------|-------|-------------|
| [dx_yolo11seg_4ch_demo](dx_yolo11seg_4ch_demo/README.md) | YOLOv11 Segmentation | Real-time instance segmentation with mask overlay across up to 4 input channels |
| [dx_yolo26_4ch_demo](dx_yolo26_4ch_demo/README.md) | YOLO26 Segmentation | Real-time segmentation with per-class BBOX toggle panel across up to 4 input channels |

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
cd dx_yolo11seg_4ch_demo

# YOLO26 Segmentation demo
cd dx_yolo26_4ch_demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure input sources

Edit the YAML config file inside `demo/config/` to set your model path and input channels (video file, RTSP stream, or camera).

### 4. Run

```bash
python -m demo.main
```

## Input Types

All demos support the following input types per channel:

| Type | Source Value | Example |
|------|-------------|---------|
| `video` | File path | `assets/videos/example.mov` |
| `rtsp` | Stream URL | `rtsp://192.168.1.100:8554/stream` |
| `camera` | Device index | `0` |

## Demo Details

- **[dx_yolo11seg_4ch_demo](dx_yolo11seg_4ch_demo/README.md)** — Uses a C++ Python binding (`dx_postprocess`) to accelerate pixel-level mask overlay operations for real-time multi-channel segmentation.
- **[dx_yolo26_4ch_demo](dx_yolo26_4ch_demo/README.md)** — Features a class list panel in the Qt GUI with per-class checkboxes to toggle BBOX display individually.
