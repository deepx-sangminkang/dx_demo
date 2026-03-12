> Korean documentation: [README-ko.md](README-ko.md)

# YOLOv11 Segmentation Multi-Channel Demo

A multi-channel Qt demo application using the YOLOv11 segmentation model.

## Screenshot

![YOLOv11 Segmentation Demo Screenshot](img/yolov11seg_4ch_demo_screenshot.png)

## Prerequisites

Before running this project, **DX-RT** (DeepX Runtime) must be built and the `dx_engine` module must be importable in Python.

```python
# The following must run without errors
import dx_engine
```

Refer to the DX-RT project documentation for installation and build instructions.

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install the dx_postprocess module

```bash
cd src/bindings/python/dx_postprocess
pip install .
cd ../../../..
```

> **Note:** `dx_postprocess` is a C++-based Python extension module that accelerates YOLO segmentation post-processing (`YOLOv8SegPostProcess`) and mask overlay (`overlay_segmentation`). Pixel-level operations that would bottleneck pure Python are handled in C++, enabling real-time multi-channel inference performance.

### 3. Download sample videos

```bash
./setup.sh
```

This downloads the sample videos used by the demo into `assets/videos/`.

## Configuration

Edit [`demo/config/yolov11_multich.yaml`](demo/config/yolov11_multich.yaml) to match your environment.

### Configuration Options

```yaml
# Model file path (DXNN format)
model: "assets/models/yolo11s-seg_optim.dxnn"

# Worker thread counts
workers:
  preprocess: 1   # Pre-processing workers
  wait: 1         # Inference wait workers
  postprocess: 2  # Post-processing workers
  draw: 1         # Rendering workers

# Input channel configuration (up to 4 channels)
channels:
  - name: "ch1"               # Channel name
    type: "video"             # Input type: video, rtsp, camera
    source: "assets/videos/example.mov"  # Input source path
    enabled: true             # Enable/disable channel
    max_fps: 25              # Maximum FPS

  - name: "ch2"
    type: "rtsp"
    source: "rtsp://192.168.1.100:8554/stream"
    enabled: true
    max_fps: 25

  - name: "ch3"
    type: "camera"
    source: 0                 # Camera device index
    enabled: false
    max_fps: 25
```

**Source value by input type:**
- `video`: Path to a video file
- `rtsp`: RTSP stream URL
- `camera`: Camera device index (0, 1, 2, ...)

## Running

```bash
python -m demo.main
```

## Project Structure

- `demo/main.py` - Qt GUI main application
- `demo/engine.py` - YOLOv11 inference engine wrapper
- `demo/workers.py` - Multi-threaded workers (capture / pre-process / post-process)
- `demo/config/yolov11_multich.yaml` - Configuration file
- `src/bindings/python/dx_postprocess/` - C++ post-processing Python bindings
- `assets/models/` - DXNN model files
- `assets/videos/` - Test video files
