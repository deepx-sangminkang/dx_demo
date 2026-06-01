# yolo26_4ch_demo 전면 개편 설계 (dx_stream / pydxs 기반)

작성일: 2026-06-01
상태: Draft (사용자 리뷰 대기)

## 1. 배경 / 문제

현재 `yolo26_4ch_demo`는 다음 구조다.

- 디코딩만 GStreamer HW(`mppvideodec`/RGA), 나머지는 Python/CPU.
- `engine.preprocess`(letterbox/resize) → `dx_engine.run_async`(NPU) → `engine.finalize_detections`(후처리) 를 Python 워커(`preprocess`/`wait`/`detect`)가 수행.
- Qt GUI(`main.py`)가 표시 + 클래스 필터 + 채널/overall FPS + paintEvent 오버레이.

RK3588(rockpi) 프로파일링 결과 `read~90`인데 `pre=inf=draw~30`으로, **CPU 전처리/후처리가 병목**이라 채널당 ~22 FPS에 묶이고 input drop이 발생한다. 호스트(x86)는 같은 코드로 ~44 FPS.

한편 dx_stream 네이티브 파이프라인(`pydxs` 기반 / `gst-launch`)은 rockpi에서 빠르게 동작한다. 이유는 전처리(RGA `dxpreprocess`)·추론(`dxinfer`)·후처리(`dxpostprocess`)가 모두 네이티브 HW 가속 GStreamer 요소이기 때문이다.

**목표**: 데모의 추론 파이프라인을 dx_stream 네이티브 요소로 교체해 CPU 병목을 제거하고, RK3588에서 채널당 FPS를 끌어올린다. 동시에 기존 Qt GUI 자산(클래스 필터·채널 통계·오버레이)을 최대한 보존한다.

## 2. dx_stream 자산 (재사용 가능, 신규 개발 거의 불필요)

조사로 확인된 기성 구성요소:

- **YOLO26 OD 완전 지원**: `dx_stream/pipelines/single_network/object_detection/run_yolo26n.sh`
  - `dxpreprocess`(preprocess-id, resize-width/height, keep_ratio, pad_value)
  - `dxinfer`(model-path=`yolo26n.dxnn`, inference-id)
  - `dxpostprocess`(`libpostprocess_yolo26od.so`, function-name=`PostProcess`)
  - `dxosd`(박스 렌더), `dxscale`(RGA 리사이즈), `compositor`+`fpsdisplaysink`(표시)
- **멀티스트림 토폴로지 2종**:
  - per-stream: `run_multi_stream.sh` — 채널마다 독립 decode→preprocess→infer→postprocess.
  - shared infer: `run_multi_stream_selector.sh` — `dxinputselector`로 모아 단일 추론 후 `dxoutputselector`로 분배(배치 효율, 16ch).
- **pydxs**: GStreamer probe에서 `pydxs.dx_get_frame_meta(hash(info))`로 `DXFrameMeta`/`DXObjectMeta`(box/label/confidence/track_id/seg) 읽기. `venv-dx_stream`에 설치됨(시스템 `python3-gi` 의존).

즉 "전처리/추론/후처리/멀티채널/YOLO26 디코더"는 이미 존재한다. 개편의 본질은 **Python 워커 파이프라인을 GStreamer 파이프라인으로 치환하고, Qt를 그 위에 얹는 글루(glue)를 만드는 것**이다.

## 3. 채택 방향: A — Qt GUI 유지 + 네이티브 추론

세 방향 중 **A**를 권장한다.

- **A (채택): Qt 유지 + 네이티브 추론.** 파이프라인을 `... ! dxpostprocess ! appsink`로 끝내고, appsink 버퍼에서 표시용 프레임을 꺼내고 같은 버퍼의 `DXFrameMeta`를 pydxs로 읽어 Qt에 전달. 기존 VideoWidget 그리드/클래스 필터/FPS/`overlay.scale_box` 오버레이를 보존. 전처리·추론·후처리는 HW로 대체되어 병목 제거.
- B (대안): 완전 네이티브 표시(`dxosd`+`compositor`+`fpsdisplaysink`). 최고 성능·최소 코드지만 인터랙티브 GUI(클래스 필터 체크박스, 채널별 통계) 상실. → "최대 성능이 필요하고 GUI 단순화를 허용"할 때의 폴백으로 문서에 병기.
- C (절충): `dxosd` 렌더 후 appsink→Qt 그리드. 네이티브 오버레이라 클래스 필터를 동적 반영하기 어려움(필터가 후처리/렌더에 고정). A보다 이점 적어 제외.

### 채택 근거
rockpi 병목은 GUI가 아니라 CPU 전처리/후처리였다. A는 그 병목만 정확히 제거하면서 이미 투자된 GUI 자산(디커플링·필터·통계)을 살린다. B 대비 약간의 글루 비용이 있으나, 데모의 가치(인터랙티브 데모 UI)를 유지한다.

## 4. 목표 아키텍처 (방향 A)

### 토폴로지: per-stream 파이프라인 × 4 (권장)
채널마다 독립 파이프라인:

```
urisourcebin/filesrc ! decodebin(HW) !
  dxpreprocess (preprocess-id=ch, resize 640, keep_ratio, pad 114) !
  dxinfer      (inference-id=1, model=yolo26n.dxnn)               !
  dxpostprocess(libpostprocess_yolo26od.so, PostProcess)         !
  [pad probe: pydxs로 DXFrameMeta 읽기] !
  appsink (drop=true, max-buffers=1, emit-signals=true)
```

- 각 appsink가 채널 고정 → 채널 식별 단순(stream_id 메타 불필요).
- `dxinfer`들은 단일 NPU를 내부적으로 공유(직렬화). 채널 확장(>8) 시 shared-infer(selector)로 전환 가능 — 토폴로지를 빌더 함수로 추상화해 교체 가능하게 설계.
- 표시 해상도: appsink 앞에 `dxscale`로 표시 크기까지 RGA 다운스케일(표시 프레임만; 검출 좌표는 `DXFrameMeta`가 프레임 좌표계로 제공) — 표시/추론 좌표계 매핑은 `overlay.scale_box` 재사용.

### 데이터 플로우
```
GStreamer 파이프라인(GLib MainLoop, 별도 스레드)
  └ appsink new-sample 콜백:
       frame(ndarray) 추출 + DXFrameMeta(objects[box/label/conf/track]) 추출
       → (channel_id, frame, detections) 를 Qt 스레드로 전달(Qt signal / thread-safe queue)
Qt 메인스레드:
  - VideoWidget[channel_id] 에 frame 갱신(표시-추론 디커플링 유지)
  - detections + selected_classes 로 paintEvent 오버레이(overlay.scale_box)
  - throughput_stats 갱신(채널별/overall FPS)
```

### GLib ↔ Qt 통합
- GStreamer는 `GLib.MainLoop`를 별도 스레드에서 실행. appsink는 `emit-signals=true` + `new-sample` 콜백(또는 폴링)으로 샘플 취득.
- 콜백→Qt 전달은 thread-safe 큐 + `QtCore.QMetaObject.invokeMethod`/signal(`Qt.QueuedConnection`)로 메인스레드 마샬링. 기존 `display_callback`/`frame_ready` signal 패턴과 동일.

## 5. 컴포넌트 매핑 (현행 → 개편)

| 현행 | 개편 후 |
|---|---|
| `engine.preprocess`(CPU letterbox) | `dxpreprocess` (RGA, keep_ratio letterbox) — 제거 |
| `engine.run_async`/`wait`(dx_engine) | `dxinfer` — 제거 |
| `engine.finalize_detections`(후처리) | `dxpostprocess`(yolo26od) + pydxs 메타 읽기 — 대폭 축소 |
| `workers.preprocess/wait/detect_worker` | 파이프라인 appsink probe 콜백 — 제거/대체 |
| `gst_pipeline.build_gst_pipeline`(디코딩만) | 추론까지 포함한 풀 파이프라인 빌더로 확장 |
| `CaptureThread` | 채널별 파이프라인 래퍼(`StreamPipeline`) |
| `main.py` Qt GUI | 거의 유지(소스: frame_ready/detections_ready 신호 입력부만 교체) |
| `overlay.scale_box` | 그대로 재사용 |
| `engine.classes`/`color_palette` | 모델/라벨 파일에서 로드(유지) |

### 클래스 필터
`DXObjectMeta.label`(class id) 기준으로 probe 또는 Qt 측에서 `selected_classes` 필터링(기존 로직 재사용). dxpostprocess 출력은 전체 클래스를 주고, 표시 단계에서 필터 → 동적 토글 유지.

## 6. 의존성 / 환경

- **pydxs 사용**: `venv-dx_stream`(`python3 -m venv --system-site-packages`)에서 실행하거나, 데모 venv에 `pydxs`를 설치하고 `python3-gi` 접근 보장. 데모 실행 진입점/문서를 이 venv 기준으로 갱신.
- **dx_stream 플러그인 설치**: `dxpreprocess/dxinfer/dxpostprocess/dxosd/dxscale` 요소가 `GST_PLUGIN_PATH`에 있어야 함(보드에 dx_stream 설치 전제).
- **모델/라벨**: `yolo26n.dxnn` + COCO 라벨. 현재 데모 모델(`yolo26n-1.dxnn`)과 후처리 라이브러리 호환성 확인 필요.
- **호스트 한계**: 호스트엔 NPU/RGA 없음 → 파이프라인 빌더 문자열·메타 파싱·필터·좌표매핑 등 **순수 로직은 단위 테스트**, 실제 추론/표시는 보드에서만 통합 검증.

## 7. 리스크 / 미해결 질문

1. **dxpostprocess 출력 포맷 [확인됨]**: `yolo26_od/postprocess.cpp`는 내부에서 **letterbox 역변환(`(box - pad)/r`, clip)** 까지 수행해 `DXObjectMeta._box`를 **원본 프레임 좌표계(`frame_meta._width × _height`)** 로 채운다. `_confidence`/`_label`/`_label_name`도 채워지고 score/NMS도 후처리 내부 처리. → demo의 `finalize_detections`/`convert_to_original_coordinates`를 **완전 대체**하며, 표시는 `overlay.scale_box`(원본→표시 pixmap)만 재사용하면 된다. (ROI가 설정되면 box에 ROI offset도 더해줌.)
2. **GLib MainLoop ↔ Qt** 안정성(스레드 마샬링, 종료 처리). 기존 디커플링 패턴으로 완화되나 실측 필요.
3. **appsink에서 frame+meta 동시 취득**: 같은 GstBuffer에서 비디오 프레임과 DXFrameMeta를 함께 꺼내는 정확한 pydxs/gi 코드 패턴 확인.
4. **채널별 max_fps/EOF 루프**: 현재 데모의 캡처 레이트 리밋·비디오 루프(EOF 재오픈)를 파이프라인에서 어떻게 재현할지(`videorate`, `urisourcebin` looping, segment seek).
5. **모델 호환**: `yolo26n.dxnn`(dx_stream 샘플) vs 데모 자산 모델의 출력 텐서/후처리 매칭.
6. **성능 목표 검증**: 개편 후 rockpi 채널당 FPS 실측(목표: 호스트 수준 또는 NPU 한계 근접). per-stream vs selector 토폴로지 비교.

## 8. 범위 / 비범위

- 범위: 추론 파이프라인 네이티브화(A), Qt 글루, 멀티채널(4), 클래스 필터·FPS·오버레이 보존, 순수 로직 단위 테스트, 보드 검증 절차.
- 비범위(차기): tracking(`dxtracker`)·segmentation·pose, 16ch selector 최적화, dx_stream 플러그인 자체 수정(현 시점 불필요), B 방향 전환.

## 9. 마이그레이션 전략

기존 코드를 한 번에 버리지 않고 **신규 경로를 플래그로 병행** 도입한다.

- config에 `engine_backend: dxstream|legacy`(기본 legacy) 추가. `dxstream`이면 네이티브 파이프라인 + pydxs 경로, `legacy`면 현행 유지.
- 단계적으로: ① 단일 채널 네이티브 파이프라인 + appsink probe PoC → ② Qt 글루(1ch 표시/오버레이) → ③ 4ch 확장 → ④ 클래스 필터/FPS 통합 → ⑤ legacy 제거.
- 각 단계는 보드에서 FPS/정상표시 검증 게이트.

(상세 단계는 writing-plans 산출물에서 작업 단위로 분해)
