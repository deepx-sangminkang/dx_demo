# YOLOv11 Segmentation 멀티 채널 데모

YOLOv11 세그멘테이션 모델을 사용한 멀티 채널 Qt 데모 애플리케이션입니다.

## 스크린샷

![YOLOv11 Segmentation 데모 스크린샷](img/yolov11seg_4ch_demo_screenshot.png)

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
2. `dx-postprocess-seg` C++ 확장 모듈이 없으면 빌드 및 설치 (세그멘테이션 후처리 및 마스크 오버레이 가속화)
3. 샘플 비디오가 없으면 다운로드 (`assets/videos/`)

데모 실행 없이 수동으로 설치하려면:

```bash
./install.sh
```

## 설정 방법

[`demo/config/yolov11_multich.yaml`](demo/config/yolov11_multich.yaml) 파일을 수정하여 환경에 맞게 설정합니다.

### 설정 요소 설명

```yaml
# 모델 파일 경로 (DXNN 형식)
model: "assets/models/yolo11s-seg_optim.dxnn"

# 워커 스레드 개수 설정
workers:
  preprocess: 1   # 전처리 워커
  wait: 1         # 추론 대기 워커
  postprocess: 2  # 후처리 워커
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

## 실행 방법

```bash
./run_demo.sh
```

## 성능 튜닝

데모 타이틀 바에 단계별 프레임 드롭 카운터가 표시됩니다. 이를 참고하여 [`demo/config/yolov11_multich.yaml`](demo/config/yolov11_multich.yaml)의 `workers:` 항목을 조정하세요.

![드롭 예시](img/drop_example_capture.png)

| 드롭 카운터 | 병목 지점 | 조치 |
|---|---|---|
| `infer drop` | 전처리가 느림 | `workers.preprocess` 증가 |
| `input drop` | 추론/대기가 느림 | `workers.wait` 증가 |
| `post drop` | 후처리가 느림 | `workers.postprocess` 증가 |
| `draw drop` | 렌더링이 느림 | `workers.draw` 증가 |

```yaml
workers:
  preprocess: 1   # infer drop이 높으면 증가
  wait: 1         # input drop이 높으면 증가
  postprocess: 2  # post drop이 높으면 증가
  draw: 1         # draw drop이 높으면 증가
```

> 최적값은 CPU 코어 수, NPU 처리량, 활성 채널 수 등 환경에 따라 다릅니다.

## 프로젝트 구조

- `demo/main.py` - Qt GUI 메인 애플리케이션
- `demo/engine.py` - YOLOv11 추론 엔진 래퍼
- `demo/workers.py` - 멀티스레드 워커 (캡처/전처리/후처리)
- `demo/config/yolov11_multich.yaml` - 설정 파일
- `src/bindings/python/dx_postprocess/` - C++ 후처리 Python 바인딩 (`dx-postprocess-seg`)
- `assets/models/` - DXNN 모델 파일
- `assets/videos/` - 테스트용 비디오 파일
