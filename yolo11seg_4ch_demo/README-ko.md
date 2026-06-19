> English documentation: [README.md](README.md)

# YOLO26 세그멘테이션 멀티채널 데모

YOLO26 인스턴스 **세그멘테이션** 모델을 사용하는 멀티채널 Qt 데모
애플리케이션으로, 전적으로 **dx_stream**(네이티브 GStreamer) 기반으로
동작합니다. **OpenCV 의존성이 없습니다.** dx_stream의 `run_yolo26n-seg.sh`
레퍼런스 파이프라인을 그대로 따르며, 세그멘테이션 마스크는 `dxosd` 엘리먼트가
하드웨어로 렌더링하고 Qt가 4개 스트림을 2x2 그리드로 합성합니다.

## 스크린샷

![YOLO26 Segmentation Demo Screenshot](img/yolo26_4ch_demo_screenshot.png)

## 아키텍처

각 채널은 **하나의 네이티브 dx_stream GStreamer 파이프라인**으로 동작합니다.

```
decodebin (HW mppvideodec)         # 하드웨어 비디오 디코딩 (VPU)
  -> dxpreprocess                  # 모델 입력용 RGA 레터박스/리사이즈
  -> dxinfer                       # NPU에서 YOLO26-seg 추론
  -> dxpostprocess                 # 세그멘테이션 디코딩 (libpostprocess_yolo26seg)
  -> [dxscale]                     # RGA 디스플레이 다운스케일 (예: 960x540)
  -> dxosd                         # 마스크/박스를 (다운스케일된) 프레임에 HW 렌더링
  -> dxconvert | videoconvert      # NV12 -> RGB (가능 시 RGA 하드웨어)
  -> appsink                       # 오버레이된 프레임을 Qt로 전달
```

추론과 세그멘테이션 오버레이는 모두 GStreamer 내부(NPU + RGA)에서 실행됩니다.
Python 측은 이미 오버레이된 작은 RGB 타일만 받아 2x2 그리드로 합성합니다. 색상
변환과 디스플레이 다운스케일은 **RGA** 하드웨어로 오프로드되고, 2x2 타일은
**Mali GPU**로 합성할 수 있으며, 각 채널은 부드러운 재생을 위해 원본 영상의
**네이티브 FPS**에
맞춰집니다.

## 사전 요구사항

이 데모는 **DX-RT**(DeepX Runtime) 위에 빌드되는 dx_stream GStreamer 플러그인
(`dxpreprocess` / `dxinfer` / `dxpostprocess` / `dxscale` / `dxconvert`)과
`pydxs` Python 바인딩을 **필수로** 요구합니다. 소프트웨어 폴백이 없으며, 누락 시
명확한 오류와 함께 시작이 중단됩니다.

설치 여부 확인:

```bash
gst-inspect-1.0 dxinfer        # dx_stream 플러그인 등록 확인
python -c "import pydxs"        # pydxs 바인딩 임포트 확인
```

둘 중 하나라도 실패하면 아래 안내에 따라 dx_stream을 설치하세요.

## 설치

데모 실행:

```bash
./run_demo.sh
```

`run_demo.sh`는 시작 전에 누락된 항목을 자동으로 확인하고 설치합니다.
1. 누락된 Python 의존성 설치 (`requirements.txt` — numpy, PySide6, PyYAML,
   packaging; **OpenCV 없음**)
2. `assets/videos/`에 샘플 영상이 없으면 다운로드

데모를 실행하지 않고 수동 설치:

```bash
./install.sh                              # 데모 전체 + dx_stream (기본)
./install.sh --skip-dxstream              # 데모만 (dx_stream 이미 설치됨)
./install.sh --dxstream-runtime-dir=PATH  # 특정 dx-runtime 체크아웃 사용
```

### dx_stream 설치

`install.sh`는 공식 DeepX
[dx-runtime](https://github.com/DEEPX-AI/dx-runtime) 설치 프로그램을 통해
기본적으로 dx_stream을 설치합니다(dx-runtime 체크아웃을 찾아 대신 실행). 수동
설치:

```bash
git clone --recurse-submodules https://github.com/DEEPX-AI/dx-runtime
cd dx-runtime
./install.sh --target=dx_stream
```

이 과정에서 dx_stream 플러그인, `pydxs`, GStreamer, json-glib, (RK3588의 경우)
`librga`가 제공됩니다. 설치되면 플러그인이 GStreamer에 등록되며(보통
`/usr/local/lib/<arch>/gstreamer-1.0` 아래), 평소처럼 데모를 실행하면 됩니다.

## 설정

환경에 맞게
[`demo/config/yolo11seg_multich.yaml`](demo/config/yolo11seg_multich.yaml)을
수정하세요.

```yaml
# 모델 파일 경로 (DXNN 포맷)
model: "assets/models/yolo26n-seg.dxnn"

# 추론 백엔드. "dxstream"만 지원됩니다(레거시 OpenCV 백엔드는 제거됨).
engine_backend: "dxstream"

dxstream:
  postprocess_library: "/usr/local/share/gstdxstream/lib/libpostprocess_yolo26seg.so"
  postprocess_function: "PostProcess"
  keep_ratio: true
  pad_value: 114

  # 디스플레이 분기의 NV12 -> RGB 색상 변환:
  #   auto : 가능 시 RGA `dxconvert`, 아니면 CPU `videoconvert` (기본)
  #   rga  : RGA `dxconvert` 강제 (없으면 경고 후 CPU로 폴백)
  #   cpu  : CPU `videoconvert` 강제
  color_convert: "auto"

  # 각 채널을 원본 영상의 네이티브 프레임레이트로 맞춤 (appsink가 버퍼 PTS에
  # 동기화). true -> 원본 fps로 부드럽게 재생되고, NPU/VPU가 실시간보다 빠르게
  # 디코딩하느라 낭비하지 않음 (기본). false -> 최대 처리량 벤치마크용 전속 실행.
  sync_to_fps: true

  # 디스플레이 다운스케일 (RGA `dxscale`): Qt로 전달되는 프레임을 이 해상도로
  # 리사이즈하여 소스 해상도와 무관하게 GUI가 작은 RGB 타일만 다루도록 함
  # (4K 입력에 필수, FHD에도 유용). 검출 박스는 dxscale 상류의 원본 프레임
  # 좌표로 읽으므로 정확함. 미설정 시 960x540 기본값. display_downscale: false로
  # 비활성화 가능.
  # display_downscale: true
  # display_width: 960
  # display_height: 540

# 코어별 최대 주파수로 자동 감지한 CPU 클러스터에 데모의 핫 스레드를 고정:
#   performance : 가장 빠른 코어 (예: RK3588 A76 cpu4-7) - 기본
#   efficiency  : 저전력 코어 (예: RK3588 A55 cpu0-3)
#   none        : CPU 어피니티 설정 안 함
cpu_affinity: "performance"

# 2x2 타일 디스플레이의 렌더링 백엔드:
#   auto : 가능 시 Mali GPU (OpenGL), 아니면 CPU QPainter (기본)
#   gpu  : GPU 렌더링 강제 (OpenGL 없으면 CPU로 폴백)
#   cpu  : CPU QPainter 렌더링 강제
render_backend: "auto"

# 입력 채널 (최대 4개)
channels:
  - name: "ch1"
    type: "video"             # video | rtsp | camera
    source: "assets/videos/cctv-city-road.mov"
    enabled: true
```

**입력 타입별 source 값:**
- `video`: 영상 파일 경로
- `rtsp`: RTSP 스트림 URL (예: `rtsp://user:pass@ip:port/stream`)
- `camera`: 카메라 장치 인덱스 (0, 1, 2, ...)

## 하드웨어 가속

RK3588(Orange Pi 5 Plus / RockPi)에서 전체 파이프라인이 하드웨어 가속됩니다.

| 단계 | 엘리먼트 | 하드웨어 |
|---|---|---|
| 비디오 디코딩 | `mppvideodec` (`decodebin`이 자동 선택) | VPU (MPP) |
| 전처리 (레터박스/리사이즈) | `dxpreprocess` | RGA |
| 추론 | `dxinfer` | NPU |
| 디스플레이 다운스케일 | `dxscale` | RGA |
| NV12 → RGB | `dxconvert` (`color_convert: auto`/`rga`) | RGA |
| 타일 합성 | `GLVideoWidget` (`render_backend: auto`/`gpu`) | Mali GPU |

협상된 디코더가 시작 시 로그로 출력되어 HW 디코딩 활성 여부를 확인할 수 있습니다
(예: `decoder: mppvideodec (HW)`). `decodebin`이 소프트웨어 디코더로 폴백하면
`(SW)`로 표시됩니다.

### 네이티브 FPS(부드러운) 재생

`sync_to_fps: true`(기본)에서는 appsink가 버퍼를 PTS 시점에 표시하므로, 각 채널이
VPU/NPU가 허용하는 최대 속도가 아니라 소스의 네이티브 프레임레이트로 재생됩니다.
백프레셔가 상류로 전파되어 디코더와 NPU가 실시간 작업만 수행 — 더 부드러운 영상과
낮은 전력. 전속 처리량 벤치마크가 필요하면 `sync_to_fps: false`로 설정하세요.

### 해상도 무관 디스플레이 (4K 지원)

디스플레이 분기는 Qt로 프레임을 넘기기 전에 항상 RGA에서 `display_width` x
`display_height`(기본 960x540)로 다운스케일하므로, GUI 비용이 소스 해상도와
무관합니다 — FHD든 4K든 모두 처리됩니다. `display_downscale: false`로 전체 해상도
프레임을 전달할 수 있습니다. 검출은 `dxscale` 상류의 원본 프레임 좌표로 읽고
오버레이가 이를 다운스케일된 타일에 매핑하므로 정확합니다.

### CPU 어피니티

시작 시 데모는
`/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`를 읽어 코어를
**efficiency**(최저 최대 주파수)와 **performance**(더 빠름) 클러스터로 분류한 뒤,
`cpu_affinity` 설정에 따라 프로세스를 고정합니다. RK3588에서는 A55 코어(cpu0-3,
약 1.8 GHz)가 efficiency 클러스터, A76 코어(cpu4-7, 약 2.25-2.3 GHz)가
performance 클러스터이며, `performance`(기본)는 A76에 고정합니다. cpufreq sysfs가
없는 플랫폼에서는 아무 동작도 하지 않습니다.

## 실행

```bash
./run_demo.sh
```

## 프로젝트 구조

- `demo/main.py` - Qt GUI 메인 애플리케이션 (CPU/GPU 비디오 위젯, 오케스트레이션)
- `demo/native_pipeline.py` - dx_stream GStreamer 실행 문자열 생성
- `demo/stream_pipeline.py` - 채널별 파이프라인 실행, Qt 브리지
- `demo/native_config.py` - 설정 로딩 / 백엔드 검증
- `demo/cpu_affinity.py` - CPU 클러스터 자동 감지 및 고정
- `demo/gst_utils.py` - GStreamer 엘리먼트 가용성 확인 (OpenCV 없음)
- `demo/meta_adapter.py` / `demo/pydxs_bridge.py` - pydxs 검출 결과 읽기
- `demo/config/yolo11seg_multich.yaml` - 설정 파일
- `assets/models/` - DXNN 모델 파일
- `assets/videos/` - 테스트 영상 파일
