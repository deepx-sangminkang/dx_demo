> Korean documentation: [README-ko.md](README-ko.md)

# YOLO26 Segmentation Multi-Channel Demo

A multi-channel Qt demo application using the YOLO26 instance **segmentation**
model, built entirely on **dx_stream** (native GStreamer). There is **no OpenCV
dependency**. It mirrors the dx_stream `run_yolo26n-seg.sh` reference pipeline:
the segmentation masks are rendered in hardware by the `dxosd` element and the Qt
front end composites the four overlaid streams into a 2x2 grid.

## Screenshot

![YOLO26 Segmentation Demo Screenshot](img/yolo26_4ch_demo_screenshot.png)

## Architecture

Each channel runs **one native dx_stream GStreamer pipeline**:

```
decodebin (HW mppvideodec)         # hardware video decode (VPU)
  -> dxpreprocess                  # RGA letterbox / resize for the model
  -> dxinfer                       # YOLO26-seg inference on the NPU
  -> dxpostprocess                 # decode segmentation (libpostprocess_yolo26seg)
  -> [dxscale]                     # RGA display downscale (e.g. 960x540)
  -> dxosd                         # HW render of masks/boxes onto the (downscaled) frame
  -> dxconvert | videoconvert      # NV12 -> RGB (RGA hardware when available)
  -> appsink                       # overlaid frames handed to Qt
```

Inference and the segmentation overlay run entirely inside GStreamer (NPU + RGA).
The Python side only receives small, already-overlaid RGB tiles and composites
the 2x2 grid. Colour conversion and display downscale are offloaded to the
**RGA** hardware, the 2x2 tiles can be composited on the **Mali GPU**, and each
channel is paced to its source video's **native FPS** for smooth playback.

## Prerequisites

This demo **requires** the dx_stream GStreamer plugin (`dxpreprocess` /
`dxinfer` / `dxpostprocess` / `dxscale` / `dxconvert`) and the `pydxs` Python
bindings, which are built on top of **DX-RT** (DeepX Runtime). There is no
software fallback — if these are missing, startup aborts with a clear error.

Verify they are present:

```bash
gst-inspect-1.0 dxinfer        # dx_stream plugin registered
python -c "import pydxs"        # pydxs bindings importable
```

If either fails, install dx_stream (see below).

## Installation

Run the demo with:

```bash
./run_demo.sh
```

`run_demo.sh` automatically:
1. Activates the dx_stream venv (installed by `install.sh --target=dx_stream`)
2. Installs any missing Python dependencies (`requirements.txt` — numpy,
   PySide6, PyYAML, packaging; **no OpenCV**)
3. Downloads sample videos into `assets/videos/` and model files if not present
4. Starts the demo

To install manually without running the demo:

```bash
./install.sh                                    # full install (dxrt + dxstream + demo)
./install.sh --skip-dxrt                        # skip NPU driver/RT/FW, install dxstream + demo
./install.sh --skip-dxstream                    # skip dxstream, install dxrt + demo
./install.sh --skip-dxrt --skip-dxstream        # demo only (both dxrt and dxstream already present)
./install.sh --runtime-dir=/path/to/dx-runtime  # use a custom dx-runtime checkout
./install.sh -f                                 # force reinstall
```

### Install flags

| Flag | Description |
|---|---|
| `--runtime-dir=PATH` | Custom dx-runtime path (default: `../dx-runtime`, auto-cloned if missing) |
| `--skip-dxrt` | Skip NPU driver/RT/FW installation (already present) |
| `--skip-dxstream` | Skip dxstream plugin + pydxs installation (already present) |
| `-f, --force` | Reinstall even if already present |

### What install.sh does

**Step 1: NPU driver / RT / FW** (unless `--skip-dxrt`)
- Checks `dxrt-cli -s`; if it fails, auto-clones dx-runtime and installs:
  - `dx_rt_npu_linux_driver`
  - `dx_rt`
  - `dx_fw`

**Step 2: dxstream + pydxs** (unless `--skip-dxstream`)
- Checks `gst-inspect-1.0 dxstream` and `pip list | grep pydxs`
- If missing, auto-clones dx-runtime and installs `dx_stream` (with `--sanity-check=n`)
- Activates venv and verifies installation

**Step 3: Python dependencies + assets**
- `pip install -r requirements.txt`
- `./setup.sh` (downloads model and sample videos)

## Configuration

Edit [`demo/config/yolo26seg_multich.yaml`](demo/config/yolo26seg_multich.yaml)
to match your environment.

```yaml
# Model file path (DXNN format)
model: "assets/models/yolo26n-seg.dxnn"

# Inference backend. Only "dxstream" is supported (the legacy OpenCV backend
# has been removed).
engine_backend: "dxstream"

dxstream:
  postprocess_library: "/usr/local/share/gstdxstream/lib/libpostprocess_yolo26seg.so"
  postprocess_function: "PostProcess"
  keep_ratio: true
  pad_value: 114

  # Render the segmentation overlay (masks + boxes) in the pipeline with the HW
  # `dxosd` element (default true). Set false to deliver clean, un-overlaid
  # frames.
  osd: true

  # NV12 -> RGB colour conversion for the display branch:
  #   auto : RGA `dxconvert` when available, else CPU `videoconvert` (default)
  #   rga  : force RGA `dxconvert` (warns + falls back to CPU if missing)
  #   cpu  : force CPU `videoconvert`
  color_convert: "auto"

  # Pace each channel to its source video's native frame rate (appsink syncs to
  # the buffer PTS). true -> smooth playback at the original fps, and the NPU/VPU
  # do not waste work decoding faster than real time (default). false -> run flat
  # out for max-throughput benchmarking.
  sync_to_fps: true

  # Display downscale (RGA `dxscale`): the frame delivered to Qt is resized to
  # this resolution, so the GUI only handles small RGB tiles regardless of the
  # source resolution (essential for 4K inputs, helpful for FHD). Detection
  # boxes stay correct because they are read upstream in original-frame coords.
  # Defaults to 960x540 when unset; set display_downscale: false to disable.
  # display_downscale: true
  # display_width: 960
  # display_height: 540

# Pin the demo's hot threads to an auto-detected CPU cluster:
#   performance : the fastest cores (e.g. RK3588 A76 cpu4-7) - default
#   efficiency  : the power-efficient cores (e.g. RK3588 A55 cpu0-3)
#   none        : do not set CPU affinity
cpu_affinity: "performance"

# Rendering backend for the 2x2 tile display:
#   auto : Mali GPU (OpenGL) when available, else CPU QPainter (default)
#   gpu  : force GPU rendering (falls back to CPU if OpenGL is unavailable)
#   cpu  : force CPU QPainter rendering
render_backend: "auto"

# Input channels (up to 4)
channels:
  - name: "ch1"
    type: "video"             # video | rtsp | camera
    source: "assets/videos/cctv-city-road.mov"
    enabled: true
```

**Source value by input type:**
- `video`: Path to a video file
- `rtsp`: RTSP stream URL (e.g. `rtsp://user:pass@ip:port/stream`)
- `camera`: Camera device index (0, 1, 2, ...)

## Hardware acceleration

The whole pipeline is hardware-accelerated on RK3588 (Orange Pi 5 Plus / RockPi):

| Stage | Element | Hardware |
|---|---|---|
| Video decode | `mppvideodec` (auto-selected by `decodebin`) | VPU (MPP) |
| Preprocess (letterbox/resize) | `dxpreprocess` | RGA |
| Inference | `dxinfer` | NPU |
| Display downscale | `dxscale` | RGA |
| NV12 → RGB | `dxconvert` (`color_convert: auto`/`rga`) | RGA |
| Tile compositing | `GLVideoWidget` (`render_backend: auto`/`gpu`) | Mali GPU |

The negotiated decoder is logged at startup so you can confirm HW decode is
active (e.g. `decoder: mppvideodec (HW)`); if `decodebin` ever falls back to a
software decoder it is reported as `(SW)`.

### Native-FPS (smooth) playback

With `sync_to_fps: true` (default), the appsink presents buffers at their PTS,
so each channel plays at its source's native frame rate rather than as fast as
the VPU/NPU allow. Backpressure propagates upstream, so the decoder and NPU only
do real-time work — smoother video and lower power. Set `sync_to_fps: false` for
flat-out throughput benchmarking.

### Resolution-agnostic display (4K ready)

The display branch always downscales to `display_width` x `display_height`
(default 960x540) on the RGA before handing frames to Qt, so the GUI cost is
independent of the source resolution — FHD or 4K inputs are both handled. Set
`display_downscale: false` to deliver full-resolution frames. Detections stay
accurate because they are read upstream of `dxscale`, in original-frame
coordinates, and the overlay maps them onto the downscaled tile.

### CPU affinity

At startup the demo reads `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`
to classify cores into **efficiency** (lowest max-freq) and **performance**
(faster) clusters, then pins the process accordingly (`cpu_affinity` config).
On RK3588 the A55 cores (cpu0-3, ~1.8 GHz) are the efficiency cluster and the
A76 cores (cpu4-7, ~2.25-2.3 GHz) are the performance cluster — `performance`
(default) pins to the A76s. It is a no-op on platforms without cpufreq sysfs.

## Running

```bash
./run_demo.sh
```

## Project Structure

- `demo/main.py` - Qt GUI main application (CPU/GPU video widgets, orchestration)
- `demo/native_pipeline.py` - Builds the dx_stream GStreamer launch string
- `demo/stream_pipeline.py` - Runs one pipeline per channel, bridges to Qt
- `demo/native_config.py` - Config loading / backend validation
- `demo/cpu_affinity.py` - CPU cluster auto-detection and pinning
- `demo/gst_utils.py` - GStreamer element availability check (no OpenCV)
- `demo/meta_adapter.py` / `demo/pydxs_bridge.py` - pydxs detection read-back
- `demo/config/yolo26seg_multich.yaml` - Configuration file
- `assets/models/` - DXNN model files
- `assets/videos/` - Test video files
