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

Run the demo with:

```bash
./run_demo.sh
```

`run_demo.sh` automatically checks and installs what is missing before starting:
1. Installs any missing Python dependencies (`requirements.txt`)
2. Builds and installs the `dx-postprocess-seg` C++ extension if not already installed
3. Downloads sample videos into `assets/videos/` if not present

To install manually without running the demo:

```bash
./install.sh
```

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
./run_demo.sh
```

## Performance Tuning

The demo shows per-stage frame drop counters in the title bar. Use them to identify bottlenecks and adjust `workers:` counts in [`demo/config/yolov11_multich.yaml`](demo/config/yolov11_multich.yaml).

![Drop example](img/drop_example_capture.png)

> The screenshot above shows `input drop` increasing — this means the preprocess workers cannot keep up with the capture rate. Increase `workers.preprocess` to resolve this.

| Drop counter | Bottleneck | Action |
|---|---|---|
| `input drop` | Preprocess workers are too slow | Increase `workers.preprocess` |
| `infer drop` | Inference wait workers are too slow | Increase `workers.wait` |
| `post drop` | Post-processing is too slow | Increase `workers.postprocess` |
| `draw drop` | Rendering is too slow | Increase `workers.draw` |

```yaml
workers:
  preprocess: 1   # increase if input drop is high
  wait: 1         # increase if infer drop is high
  postprocess: 2  # increase if post drop is high
  draw: 1         # increase if draw drop is high
```

> Optimal values depend on your hardware (CPU cores, NPU throughput, number of active channels).

## Project Structure

- `demo/main.py` - Qt GUI main application
- `demo/engine.py` - YOLOv11 inference engine wrapper
- `demo/workers.py` - Multi-threaded workers (capture / pre-process / post-process)
- `demo/config/yolov11_multich.yaml` - Configuration file
- `src/bindings/python/dx_postprocess/` - C++ post-processing Python bindings (`dx-postprocess-seg`)
- `assets/models/` - DXNN model files
- `assets/videos/` - Test video files
