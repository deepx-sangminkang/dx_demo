# yolo_multi_demo

`yolo_multi_demo` is a multi-channel YOLO object detection demo that processes multiple input channels simultaneously.

## Overview

- Executable: `bin/yolo_multi_demo`
- Runs from configuration files
- Supports multi-channel object detection scenarios

At launch you can select one of the following configurations:

- `0`: `config/yolo_multi_demo.json`
- `1`: `config/ppu_yolo_multi_demo.json`
- `2`: `config/ppu_yolo_multi_100channel_demo.json`

## How to Run

```bash
cd yolo_multi_demo
./run_demo.sh
```

## What the Run Script Does

`run_demo.sh` handles the following automatically:

1. Runs `install.sh` if `bin/yolo_multi_demo` is missing.
2. Runs `setup.sh` if `assets/models` or `assets/videos` is missing.
3. Passes the configuration file selected from the menu via the `-c` option and starts the demo.

If no input is given, the default value `0` is selected.

## How to Exit

- Press `ESC` or `Q` while running to exit.

## Camera Expand Layout

When `video_sources` includes a camera input (one of `"camera"`, `"camera_image"`, `"camera_video"`), the first camera channel is automatically **expanded into the center of the grid** to emphasize it like a main window. The remaining channels are placed in the surrounding cells as usual.

- The expanded area size is automatically computed to stay within about 20% of the total grid area, and is progressively shrunk if there are not enough cells left for the other inputs (`scale ≤ floor(sqrt(0.20 × cols × rows))`).
- A **yellow border** is drawn around every cell containing a camera input for visual distinction.
- This does not apply when `display_config.expand_mode` (for special channel counts 33/41/61/73) is enabled.

## Additional display_config Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sidebar_font_scale` | float | 1.0 | Sidebar font scale |
| `fps_value_font_scale` | float | 0.5 | Top HUD FPS value font scale |

Example (part of `video_sources`):

```json
"video_sources": [
    [ "0", "camera" ],
    [ "./assets/videos/cctv-city-road.mov", "offline", 60 ],
    [ "./assets/videos/dance-group.mov",    "offline", 60 ]
]
```

## Screenshots

**0: Multi Channel Object Detection**

![yolo_multi_demo](img/yolo_multi_demo_screenshot.png)

**1: Multi Channel Object Detection With PPU**

![yolo_multi_demo_with_ppu](img/yolo_multi_demo_with_ppu_screenshot.png)

**2: Multi Channel 100ch Object Detection With PPU**

![yolo_multi_demo_100channel_with_ppu](img/yolo_multi_demo_100channel_with_ppu_screenshot.png)
