# YOLO26 멀티 채널 데모

YOLO26 검출 모델을 사용한 멀티 채널 Qt 데모 애플리케이션입니다.

- Qt GUI 우측에 **클래스 리스트 패널**이 있으며, 각 클래스의 체크 박스를 통해
  해당 클래스에 대한 BBOX 표시 여부를 개별적으로 제어할 수 있습니다.

## 스크린샷

![YOLO26 데모 스크린샷](img/yolo26_4ch_demo_screenshot.png)

## 사전 요구사항

이 프로젝트를 실행하기 전에 **DX-RT**(DeepX Runtime)가 빌드되어 있어야 하며, Python에서 `dx_engine` 모듈을 import할 수 있어야 합니다.

```python
# 다음 명령이 오류 없이 실행되어야 합니다
import dx_engine
```

DX-RT 설치 및 빌드 방법은 해당 프로젝트의 문서를 참고하세요.

## 설치 방법

다음 명령으로 데모를 실행하세요:

```bash
./run_demo.sh
```

`run_demo.sh`는 실행 전 누락된 항목을 자동으로 확인하고 설치합니다:
1. 누락된 Python 의존성 패키지 설치 (`requirements.txt`)
2. 샘플 비디오가 없으면 다운로드 (`assets/videos/`)

데모 실행 없이 수동으로 설치하려면:

```bash
./install.sh
```

## 설정 방법

[`demo/config/yolo26_multich.yaml`](demo/config/yolo26_multich.yaml) 파일을 수정하여 환경에 맞게 설정합니다.

### 설정 요소 설명

```yaml
# 모델 파일 경로 (DXNN 형식)
model: "assets/models/yolo11s-seg_optim.dxnn"

# 비디오 디코딩 모드 (전역 기본값, 채널별로 재정의 가능)
#   auto : 지원되는 환경에서 HW 디코딩, 아니면 SW (기본값)
#   hw   : HW 디코딩 강제 (불가 시 SW로 폴백)
#   sw   : SW 디코딩 강제
decode: "auto"

# 워커 스레드 개수 설정
workers:
  preprocess: 1   # 전처리 워커
  wait: 1         # 추론 대기 워커
  draw: 1         # 렌더링 워커

# 입력 채널 설정 (최대 4개)
channels:
  - name: "ch1"               # 채널 이름
    type: "video"             # 입력 타입: video, rtsp, camera
    source: "assets/videos/example.mov"  # 입력 소스 경로
    enabled: true             # 채널 활성화 여부
    max_fps: 25              # 최대 FPS

  - name: "ch2"
    type: "rtsp"
    source: "rtsp://192.168.1.100:8554/stream"
    enabled: true
    max_fps: 25

  - name: "ch3"
    type: "camera"
    source: 0                 # 카메라 장치 번호
    enabled: false
    max_fps: 25
```

**입력 타입별 source 설정:**
- `video`: 비디오 파일 경로
- `rtsp`: RTSP 스트림 URL
- `camera`: 카메라 장치 번호 (0, 1, 2, ...)

## 하드웨어 가속 디코딩 (GStreamer)

기본적으로 각 채널은 CPU(소프트웨어)로 비디오를 디코딩합니다. `decode: "auto"`(또는 `"hw"`)로
설정하면 GStreamer 파이프라인(`cv2.VideoCapture(..., cv2.CAP_GSTREAMER)`)을 통해 디코딩을
플랫폼 하드웨어 디코더로 오프로드하여, 다채널·고해상도 환경에서 CPU 부하를 줄입니다.

**플랫폼은 자동 감지됩니다:**

| 플랫폼 | HW 디코더 | 필요 플러그인 |
|---|---|---|
| RK3588 (Orange Pi 5 Plus) | `mppvideodec` | 공식 Rockchip 이미지에 기본 포함 |
| Intel iGPU | VAAPI (`vaapidecodebin`) | `sudo apt install gstreamer1.0-vaapi` |
| NVIDIA | `nvh264dec` / `nvv4l2decoder` | NVIDIA GStreamer / DeepStream 플러그인 |

**사전 요구사항:**

1. **OpenCV가 GStreamer 지원으로 빌드**되어 있어야 합니다. PyPI `opencv-python` 휠은
   GStreamer **미지원** 빌드입니다. 다음으로 확인하세요:
   ```bash
   python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
   ```
   `GStreamer: NO`로 표시되면 GStreamer 지원 OpenCV를 설치하세요(RK3588 시스템 이미지는
   이미 제공함. 그 외 플랫폼은 배포판 `python3-opencv` 패키지 또는 커스텀 빌드 사용).
2. 위 표의 플랫폼별 디코더 플러그인을 설치하세요.

두 조건 중 하나라도 충족되지 않으면 데모는 자동으로 **소프트웨어 디코딩으로 폴백**하며,
시작 시 채널별로 이유를 출력합니다. 예:

```
[INFO] Channel 0: decode=SW (video) - OpenCV built without GStreamer support; using SW decode
```

> **성능 참고:** HW 디코딩은 주로 CPU 사용량을 줄여줍니다. 추론/드로잉을 위해 프레임이
> 결국 CPU 메모리에 BGR로 돌아와야 하므로 GPU→CPU 복사 비용은 남으며, 채널 수가 많거나
> 고해상도일수록 효과가 큽니다.

## 실행 방법

```bash
./run_demo.sh
```

## 성능 튜닝

데모 타이틀 바에 단계별 프레임 드롭 카운터가 표시됩니다. 이를 참고하여 [`demo/config/yolo26_multich.yaml`](demo/config/yolo26_multich.yaml)의 `workers:` 항목을 조정하세요.

![드롭 예시](img/drop_example_capture.png)

> 위 스크린샷은 `input drop`이 증가하는 예시입니다. 전처리 워커가 캡처 속도를 따라가지 못하는 상황으로, `workers.preprocess`를 늘려 해결할 수 있습니다.

| 드롭 카운터 | 병목 지점 | 조치 |
|---|---|---|
| `input drop` | 전처리 워커가 느림 | `workers.preprocess` 증가 |
| `infer drop` | 추론 대기 워커가 느림 | `workers.wait` 증가 |
| `draw drop` | 렌더링이 느림 | `workers.draw` 증가 |

```yaml
workers:
  preprocess: 1   # input drop이 높으면 증가
  wait: 1         # infer drop이 높으면 증가
  draw: 1         # draw drop이 높으면 증가
```

> 최적값은 CPU 코어 수, NPU 처리량, 활성 채널 수 등 환경에 따라 다릅니다.

## 프로젝트 구조

- `demo/main.py` - Qt GUI 메인 애플리케이션
- `demo/engine.py` - YOLO26 추론 엔진 래퍼
- `demo/workers.py` - 멀티스레드 워커 (캡처/전처리/후처리)
- `demo/config/yolo26_multich.yaml` - 설정 파일
- `assets/models/` - DXNN 모델 파일
- `assets/videos/` - 테스트용 비디오 파일
