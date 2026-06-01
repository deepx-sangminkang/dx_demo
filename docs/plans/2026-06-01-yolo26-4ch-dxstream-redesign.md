# yolo26_4ch_demo dx_stream/pydxs 전면 개편 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `yolo26_4ch_demo`의 추론 파이프라인을 dx_stream 네이티브 GStreamer 요소(`dxpreprocess`/`dxinfer`/`dxpostprocess`)로 교체하고 검출을 `pydxs` frame-meta probe로 읽어, RK3588의 CPU 전처리/후처리 병목을 제거하면서 기존 Qt GUI(채널 그리드·클래스 필터·FPS·오버레이)를 보존한다.

**Architecture:** 채널마다 독립 GStreamer 파이프라인 `decode → dxpreprocess → dxinfer → dxpostprocess → appsink`을 별도 GLib MainLoop 스레드에서 구동한다. appsink `new-sample` 콜백에서 표시 프레임(ndarray)과 같은 버퍼의 `DXFrameMeta`(box/label/confidence)를 추출해 기존 `frame_ready`/`detections_ready` Qt 시그널로 흘려보낸다. dxpostprocess가 이미 원본 좌표계로 letterbox 역변환을 수행하므로 표시 경로(`overlay.scale_box`)는 무수정 재사용한다. 신규 경로는 config `engine_backend: dxstream|legacy` 플래그로 기존 경로와 병행 도입 후 단계적으로 legacy를 제거한다.

**Tech Stack:** Python 3, PySide/PyQt(Qt), GStreamer 1.0(`gi`), dx_stream 플러그인(`dxpreprocess`/`dxinfer`/`dxpostprocess`/`dxscale`), `pydxs`, NumPy, pytest.

**참고 설계 문서:** `docs/plans/2026-06-01-yolo26-4ch-dxstream-redesign-design.md`

---

## 사전 메모 (실행 전 반드시 읽을 것)

- **검증 환경 분리:** 현재 호스트엔 NPU/RGA/dx_stream 플러그인/pydxs가 없을 수 있다. 따라서 이 플랜은 두 부류로 나뉜다.
  - **호스트 TDD 가능(Phase 1~3 대부분):** 파이프라인 문자열 빌더, `DXFrameMeta→detections` 어댑터, 클래스 필터, 좌표 매핑 등 **순수 로직** — pytest로 완전 검증.
  - **보드 통합 전용(Phase 4~6):** 실제 GStreamer 구동/추론/표시 — RK3588(rockpi)에서 수동 검증. 각 보드 단계는 "Manual verification" 게이트로 표기.
- **pydxs/gi 의존:** 실행은 `venv-dx_stream`(system-site-packages, `python3-gi` 접근) 기준. import 실패 시 graceful 경고 + legacy 폴백.
- **현행 자산 재사용:** `overlay.scale_box`, `throughput_stats`(`get_fps`), `VideoWidget`, 클래스 필터 패널, `frame_ready(int,object,dict)`/`detections_ready(int,object,dict)` 시그널 그대로 사용.
- **detections 포맷 계약:** 기존 `on_detections_ready`가 받는 detections는 `np.ndarray` shape `(N,6)` = `[x1,y1,x2,y2,score,class_id]` (원본 프레임 픽셀 좌표). 신규 어댑터도 **반드시 동일 포맷**으로 만들어 표시 경로를 무수정 재사용한다.
- 각 Task는 2~5분 단위. TDD(실패 테스트 → 최소 구현 → 통과 → 커밋). 자주 커밋.

---

## Phase 1: 네이티브 파이프라인 문자열 빌더 (호스트 TDD)

기존 `demo/gst_pipeline.py`는 "디코딩만" 파이프라인을 만든다. 추론까지 포함한 풀 파이프라인 빌더를 **새 모듈**로 추가한다(기존 빌더는 legacy 경로가 계속 사용하므로 보존).

### Task 1: 네이티브 파이프라인 빌더 모듈 골격

**Files:**
- Create: `demo/native_pipeline.py`
- Test: `tests/test_native_pipeline.py`

**Step 1: 실패 테스트 작성** — `tests/test_native_pipeline.py`

```python
from demo import native_pipeline as npl


def test_build_infer_pipeline_video_contains_core_elements():
    p = npl.build_infer_pipeline(
        source_type="video",
        source="/data/a.mp4",
        preprocess_cfg=npl.PreprocessCfg(preprocess_id=1, width=640, height=640,
                                         keep_ratio=True, pad_value=114),
        infer_cfg=npl.InferCfg(inference_id=1, model_path="/m/yolo26n.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(
            inference_id=1,
            library_file_path="/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so",
            function_name="PostProcess"),
        appsink_name="sink0",
    )
    assert "decodebin" in p
    assert "dxpreprocess" in p and "resize-width=640" in p and "resize-height=640" in p
    assert "dxinfer" in p and "model-path=/m/yolo26n.dxnn" in p
    assert "dxpostprocess" in p and "libpostprocess_yolo26od.so" in p
    assert "function-name=PostProcess" in p
    assert "appsink name=sink0" in p
    # appsink는 최신 프레임만, 논블로킹
    assert "drop=true" in p and "max-buffers=1" in p and "sync=false" in p
    # 추론 요소 순서: preprocess < infer < postprocess < appsink
    assert p.index("dxpreprocess") < p.index("dxinfer") < p.index("dxpostprocess") < p.index("appsink")
```

**Step 2: 실패 확인**

Run: `cd yolo26_4ch_demo && python -m pytest tests/test_native_pipeline.py -q`
Expected: FAIL (ModuleNotFoundError: demo.native_pipeline)

**Step 3: 최소 구현** — `demo/native_pipeline.py`

```python
"""Native dx_stream inference pipeline string builder.

Builds GStreamer launch strings that run the full preprocess->infer->postprocess
chain on dx_stream HW elements, terminating in an appsink so a Python/Qt front
end can read decoded frames and detection metadata (via pydxs).

Pure string construction so it is unit-testable without GStreamer present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

_APPSINK = "appsink drop=true max-buffers=1 sync=false emit-signals=true"


@dataclass
class PreprocessCfg:
    preprocess_id: int = 1
    width: int = 640
    height: int = 640
    keep_ratio: bool = True
    pad_value: int = 114


@dataclass
class InferCfg:
    inference_id: int = 1
    model_path: str = ""


@dataclass
class PostprocessCfg:
    inference_id: int = 1
    library_file_path: str = ""
    function_name: str = "PostProcess"


def _preprocess_element(cfg: PreprocessCfg) -> str:
    return (
        f"dxpreprocess preprocess-id={cfg.preprocess_id} "
        f"resize-width={cfg.width} resize-height={cfg.height} "
        f"keep-ratio={'true' if cfg.keep_ratio else 'false'} "
        f"pad-value={cfg.pad_value}"
    )


def _infer_element(cfg: InferCfg, preprocess_id: int) -> str:
    return (
        f"dxinfer preprocess-id={preprocess_id} "
        f"inference-id={cfg.inference_id} model-path={cfg.model_path}"
    )


def _postprocess_element(cfg: PostprocessCfg) -> str:
    return (
        f"dxpostprocess inference-id={cfg.inference_id} "
        f"library-file-path={cfg.library_file_path} "
        f"function-name={cfg.function_name}"
    )


def build_infer_pipeline(
    source_type: str,
    source: Union[int, str],
    preprocess_cfg: PreprocessCfg,
    infer_cfg: InferCfg,
    postprocess_cfg: PostprocessCfg,
    appsink_name: str,
    display_size: Optional[Tuple[int, int]] = None,
) -> str:
    q = "queue max-size-buffers=1"
    pre = _preprocess_element(preprocess_cfg)
    inf = _infer_element(infer_cfg, preprocess_cfg.preprocess_id)
    post = _postprocess_element(postprocess_cfg)

    if source_type == "video":
        src = f"filesrc location={source} ! decodebin"
    elif source_type == "rtsp":
        src = (
            f"rtspsrc location={source} latency=100 ! "
            f"rtph264depay ! h264parse ! decodebin"
        )
    elif source_type == "camera":
        dev = source if str(source).startswith("/dev/") else f"/dev/video{source}"
        src = f"v4l2src device={dev}"
    else:
        raise ValueError(f"unsupported source type: {source_type!r}")

    # Optional RGA downscale for the *display* branch (detections stay in
    # original frame coords because dxpostprocess already de-letterboxes).
    tail = ""
    if display_size is not None:
        w, h = display_size
        tail = f" ! dxscale width={w} height={h}"

    return (
        f"{src} ! {q} ! {pre} ! {q} ! {inf} ! {q} ! {post}{tail} ! {q} ! "
        f"{_APPSINK} name={appsink_name}"
    )
```

**Step 4: 통과 확인**

Run: `python -m pytest tests/test_native_pipeline.py -q`
Expected: PASS

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/native_pipeline.py yolo26_4ch_demo/tests/test_native_pipeline.py
git commit -m "feat: add native dx_stream inference pipeline string builder"
```

### Task 2: rtsp/camera 소스 + display_size 분기 테스트

**Files:**
- Modify: `tests/test_native_pipeline.py`

**Step 1: 실패 테스트 추가**

```python
def test_build_infer_pipeline_rtsp_and_display_scale():
    p = npl.build_infer_pipeline(
        source_type="rtsp", source="rtsp://10.0.0.1/s",
        preprocess_cfg=npl.PreprocessCfg(),
        infer_cfg=npl.InferCfg(model_path="/m/y.dxnn"),
        postprocess_cfg=npl.PostprocessCfg(library_file_path="/l.so"),
        appsink_name="s1", display_size=(320, 240),
    )
    assert "rtspsrc location=rtsp://10.0.0.1/s" in p
    assert "dxscale width=320 height=240" in p
    # dxscale은 표시 분기이므로 postprocess 이후
    assert p.index("dxpostprocess") < p.index("dxscale") < p.index("appsink")


def test_build_infer_pipeline_invalid_source_raises():
    import pytest
    with pytest.raises(ValueError):
        npl.build_infer_pipeline(
            source_type="bogus", source="x",
            preprocess_cfg=npl.PreprocessCfg(),
            infer_cfg=npl.InferCfg(), postprocess_cfg=npl.PostprocessCfg(),
            appsink_name="s")
```

**Step 2:** Run `pytest tests/test_native_pipeline.py -q` → 새 2개 PASS (구현이 이미 커버) 또는 미세 수정.
**Step 3~4:** 필요 시 `build_infer_pipeline` 보강 후 통과.
**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/tests/test_native_pipeline.py
git commit -m "test: cover rtsp/camera/display-scale/invalid-source in native pipeline"
```

---

## Phase 2: DXFrameMeta → detections 어댑터 (호스트 TDD)

`pydxs`가 없는 호스트에서도 로직을 테스트하기 위해, **메타 객체를 추상화**한 어댑터를 만든다. 입력은 "objects를 iterate하면 `.box=[x1,y1,x2,y2]`, `.label`, `.confidence`를 주는 객체"이고, 출력은 `(N,6)` ndarray.

### Task 3: detections 어댑터

**Files:**
- Create: `demo/meta_adapter.py`
- Test: `tests/test_meta_adapter.py`

**Step 1: 실패 테스트** — `tests/test_meta_adapter.py`

```python
import numpy as np
from demo import meta_adapter as ma


class _FakeObj:
    def __init__(self, box, label, conf):
        self.box = box
        self.label = label
        self.confidence = conf


class _FakeFrameMeta:
    def __init__(self, w, h, objs):
        self.width = w
        self.height = h
        self._objs = objs

    def __iter__(self):
        return iter(self._objs)


def test_frame_meta_to_detections_basic():
    fm = _FakeFrameMeta(1920, 1080, [
        _FakeObj([10, 20, 110, 220], 0, 0.9),
        _FakeObj([5, 5, 50, 60], 2, 0.5),
    ])
    det = ma.frame_meta_to_detections(fm)
    assert isinstance(det, np.ndarray)
    assert det.shape == (2, 6)
    assert det.dtype == np.float32
    np.testing.assert_allclose(det[0], [10, 20, 110, 220, 0.9, 0], rtol=1e-6)
    assert det[1, 5] == 2


def test_frame_meta_to_detections_empty():
    fm = _FakeFrameMeta(640, 480, [])
    det = ma.frame_meta_to_detections(fm)
    assert det.shape == (0, 6)


def test_filter_by_classes_none_returns_all():
    det = np.array([[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.8, 5]], np.float32)
    assert ma.filter_by_classes(det, None).shape == (2, 6)


def test_filter_by_classes_subset():
    det = np.array([[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.8, 5]], np.float32)
    out = ma.filter_by_classes(det, {5})
    assert out.shape == (1, 6) and out[0, 5] == 5


def test_filter_by_classes_empty_set_returns_empty():
    det = np.array([[0, 0, 1, 1, 0.9, 0]], np.float32)
    assert ma.filter_by_classes(det, set()).shape == (0, 6)
```

**Step 2: 실패 확인**

Run: `python -m pytest tests/test_meta_adapter.py -q`
Expected: FAIL (ModuleNotFoundError)

**Step 3: 최소 구현** — `demo/meta_adapter.py`

```python
"""Adapters from dx_stream DXFrameMeta to the demo's (N,6) detection array.

Keeps pydxs/GStreamer out of the import path so the conversion + class filter
logic is unit-testable on any host. The frame-meta argument only needs to be
iterable over objects exposing ``box``/``label``/``confidence``.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np


def frame_meta_to_detections(frame_meta) -> np.ndarray:
    """Convert a DXFrameMeta-like object to ``(N,6)`` ``[x1,y1,x2,y2,score,cls]``.

    Boxes are already in original-frame pixel coordinates (dxpostprocess
    de-letterboxes), so no rescale is applied here.
    """
    rows = []
    for obj in frame_meta:
        x1, y1, x2, y2 = obj.box[0], obj.box[1], obj.box[2], obj.box[3]
        rows.append((x1, y1, x2, y2, float(obj.confidence), float(obj.label)))
    if not rows:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def filter_by_classes(
    detections: np.ndarray, selected: Optional[Set[int]]
) -> np.ndarray:
    """Keep only rows whose class id is in ``selected``.

    ``None`` keeps everything (no filter active); empty set keeps nothing.
    """
    if selected is None:
        return detections
    if len(detections) == 0:
        return detections
    if len(selected) == 0:
        return np.empty((0, 6), dtype=detections.dtype)
    mask = np.isin(detections[:, 5].astype(int), list(selected))
    return detections[mask]
```

**Step 4: 통과 확인**

Run: `python -m pytest tests/test_meta_adapter.py -q`
Expected: PASS (5 passed)

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/meta_adapter.py yolo26_4ch_demo/tests/test_meta_adapter.py
git commit -m "feat: add DXFrameMeta->detections adapter and class filter"
```

---

## Phase 3: pydxs 브리지 + config/backend 플래그 (호스트 TDD: import 가드/플래그)

### Task 4: pydxs 가용성 가드

appsink 버퍼에서 `DXFrameMeta`를 읽는 코드는 `pydxs`/`gi` 의존이라 호스트에서 import 실패할 수 있다. 가용성 헬퍼로 감싸 graceful 처리한다.

**Files:**
- Create: `demo/pydxs_bridge.py`
- Test: `tests/test_pydxs_bridge.py`

**Step 1: 실패 테스트** — `tests/test_pydxs_bridge.py`

```python
from demo import pydxs_bridge as pb


def test_pydxs_available_returns_bool():
    assert isinstance(pb.pydxs_available(), bool)


def test_read_detections_returns_empty_when_pydxs_missing(monkeypatch):
    # When pydxs is not importable, the bridge must degrade to "no detections"
    # rather than raising, so the display path keeps running.
    monkeypatch.setattr(pb, "_PYDXS", None, raising=False)
    det = pb.read_detections_from_buffer(buffer_addr=0, width=640, height=480)
    assert det.shape == (0, 6)
```

**Step 2: 실패 확인** → FAIL (ModuleNotFoundError)

**Step 3: 최소 구현** — `demo/pydxs_bridge.py`

```python
"""Thin bridge that reads detections from a GstBuffer via pydxs.

Isolated so the rest of the demo can import it unconditionally; when pydxs is
unavailable (e.g. dev host without dx_stream) it degrades to empty detections.
"""

from __future__ import annotations

import numpy as np

from demo.meta_adapter import frame_meta_to_detections

try:  # pragma: no cover - environment dependent
    import pydxs as _PYDXS
except Exception:  # noqa: BLE001
    _PYDXS = None


def pydxs_available() -> bool:
    return _PYDXS is not None


def read_detections_from_buffer(buffer_addr: int, width: int, height: int) -> np.ndarray:
    """Read DXFrameMeta from a buffer address and return (N,6) detections.

    Returns an empty (0,6) array when pydxs is unavailable or no meta exists.
    """
    if _PYDXS is None:
        return np.empty((0, 6), dtype=np.float32)
    frame_meta = _PYDXS.dx_get_frame_meta(buffer_addr)
    if frame_meta is None:
        return np.empty((0, 6), dtype=np.float32)
    return frame_meta_to_detections(frame_meta)
```

**Step 4: 통과 확인** → PASS

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/pydxs_bridge.py yolo26_4ch_demo/tests/test_pydxs_bridge.py
git commit -m "feat: add pydxs bridge with graceful host fallback"
```

### Task 5: config 스키마에 engine_backend + dxstream 섹션

**Files:**
- Modify: `demo/config/yolo26_multich.yaml`
- Create: `tests/test_config_backend.py`
- (필요 시) Modify: `demo/main.py`의 config 로더가 신규 키를 무시하지 않도록 — 단순 dict 접근이라 변경 최소.

**Step 1: 실패 테스트** — `tests/test_config_backend.py`

```python
import yaml
from pathlib import Path


def test_config_has_backend_and_dxstream_block():
    cfg = yaml.safe_load(
        Path("demo/config/yolo26_multich.yaml").read_text()
    )
    # default must stay legacy so existing behaviour is unchanged until opt-in
    assert cfg.get("engine_backend", "legacy") in ("legacy", "dxstream")
    dxs = cfg.get("dxstream", {})
    assert "postprocess_library" in dxs
    assert "postprocess_function" in dxs
```

**Step 2: 실패 확인** → FAIL (KeyError/None)

**Step 3: 최소 구현** — `demo/config/yolo26_multich.yaml` 상단에 추가

```yaml
# Inference backend:
#   legacy   : Python preprocess + dx_engine run_async + Python postprocess (current)
#   dxstream : native GStreamer dxpreprocess/dxinfer/dxpostprocess + pydxs meta
engine_backend: "legacy"

# Native (dxstream) backend settings; used only when engine_backend == dxstream.
dxstream:
  postprocess_library: "/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so"
  postprocess_function: "PostProcess"
  preprocess_id: 1
  inference_id: 1
  keep_ratio: true
  pad_value: 114
```

**Step 4: 통과 확인** → PASS

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/config/yolo26_multich.yaml yolo26_4ch_demo/tests/test_config_backend.py
git commit -m "feat: add engine_backend flag and dxstream config block (default legacy)"
```

---

## Phase 4: 스트림 파이프라인 런너 (보드 통합)

호스트에서 구조/시그널 배선은 단위 테스트하되, 실제 GStreamer 구동은 보드 검증.

### Task 6: StreamPipeline 클래스 (배선은 호스트 테스트, 구동은 보드)

**Files:**
- Create: `demo/stream_pipeline.py`
- Test: `tests/test_stream_pipeline.py` (gi를 모킹해 콜백→시그널 배선만 검증)

**Step 1: 실패 테스트** — gi/Gst를 주입 가능하게 설계하여 모킹.

```python
import numpy as np
from demo import stream_pipeline as sp


def test_on_sample_emits_frame_and_detections(monkeypatch):
    # Simulate a decoded sample: a (H,W,3) frame + a fake buffer addr.
    frame = np.zeros((480, 640, 3), np.uint8)
    captured = {}

    def fake_extract(sample):
        return frame, 12345  # (ndarray, buffer_addr)

    def fake_read_det(buffer_addr, width, height):
        assert buffer_addr == 12345
        return np.array([[1, 2, 3, 4, 0.9, 0]], np.float32)

    runner = sp.StreamPipeline(
        channel_id=2, pipeline_str="fakepipe", appsink_name="s2",
        on_frame=lambda ch, f: captured.setdefault("frame", (ch, f)),
        on_detections=lambda ch, d, m: captured.setdefault("det", (ch, d, m)),
        sample_extractor=fake_extract,
        detection_reader=fake_read_det,
    )
    runner._handle_sample("fake_sample")  # internal callback under test

    assert captured["frame"][0] == 2
    assert captured["det"][0] == 2
    assert captured["det"][1].shape == (1, 6)
    assert captured["det"][2]["color_format"] == "rgb"
```

**Step 2: 실패 확인** → FAIL

**Step 3: 최소 구현** — `demo/stream_pipeline.py` (gi import는 lazy/주입)

```python
"""Per-channel native GStreamer pipeline runner.

Owns one inference pipeline + appsink, runs it on a GLib MainLoop thread, and
on each decoded sample emits (frame) and (detections) through callbacks that
the Qt layer wires to its frame_ready/detections_ready signals.

GStreamer access is injected (sample_extractor/detection_reader) so the
callback wiring is unit-testable without gi present.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np


class StreamPipeline:
    def __init__(
        self,
        channel_id: int,
        pipeline_str: str,
        appsink_name: str,
        on_frame: Callable[[int, np.ndarray], None],
        on_detections: Callable[[int, np.ndarray, dict], None],
        sample_extractor: Optional[Callable] = None,
        detection_reader: Optional[Callable] = None,
    ) -> None:
        self.channel_id = channel_id
        self.pipeline_str = pipeline_str
        self.appsink_name = appsink_name
        self._on_frame = on_frame
        self._on_detections = on_detections
        self._extract = sample_extractor or self._default_extract
        self._read_det = detection_reader or self._default_read_det
        self._pipeline = None
        self._loop = None
        self._thread: Optional[threading.Thread] = None

    # --- sample handling (unit-tested) ---
    def _handle_sample(self, sample) -> None:
        frame, buffer_addr = self._extract(sample)
        h, w = frame.shape[:2]
        meta = {"color_format": "rgb"}
        self._on_frame(self.channel_id, frame)
        det = self._read_det(buffer_addr, w, h)
        self._on_detections(self.channel_id, det, meta)

    # --- defaults bind to gi/pydxs at runtime (board only) ---
    def _default_extract(self, sample):  # pragma: no cover - needs gi
        from demo._gst_sample import sample_to_frame_and_addr
        return sample_to_frame_and_addr(sample)

    def _default_read_det(self, buffer_addr, w, h):  # pragma: no cover - needs pydxs
        from demo.pydxs_bridge import read_detections_from_buffer
        return read_detections_from_buffer(buffer_addr, w, h)

    # --- lifecycle (board only) ---
    def start(self) -> None:  # pragma: no cover - needs gi
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib

        Gst.init(None)
        self._pipeline = Gst.parse_launch(self.pipeline_str)
        appsink = self._pipeline.get_by_name(self.appsink_name)
        appsink.connect("new-sample", self._on_new_sample_gst)
        self._loop = GLib.MainLoop()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

    def _on_new_sample_gst(self, appsink):  # pragma: no cover - needs gi
        from gi.repository import Gst
        sample = appsink.emit("pull-sample")
        if sample is not None:
            self._handle_sample(sample)
        return Gst.FlowReturn.OK

    def stop(self) -> None:  # pragma: no cover - needs gi
        from gi.repository import Gst
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()
```

**Step 4: 통과 확인** → PASS (배선 테스트)

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/stream_pipeline.py yolo26_4ch_demo/tests/test_stream_pipeline.py
git commit -m "feat: add per-channel StreamPipeline with injectable gst access"
```

### Task 7: GstSample → (frame, buffer_addr) 추출기 (보드 전용 구현 + 호스트 스모크)

**Files:**
- Create: `demo/_gst_sample.py`
- Test: `tests/test_gst_sample_import.py` (import 가능성/시그니처만; 실동작은 보드)

**Step 1: 실패 테스트**

```python
def test_module_exposes_extractor():
    from demo import _gst_sample
    assert hasattr(_gst_sample, "sample_to_frame_and_addr")
```

**Step 2: 실패 확인** → FAIL

**Step 3: 최소 구현** — `demo/_gst_sample.py`

```python
"""GstSample -> (numpy frame, buffer address) extraction (board runtime).

Reads caps for width/height/format, maps the buffer, and wraps it as an RGB
ndarray. The integer buffer address is what pydxs uses to fetch DXFrameMeta.
"""

from __future__ import annotations

import numpy as np


def sample_to_frame_and_addr(sample):  # pragma: no cover - needs gi at runtime
    from gi.repository import Gst  # noqa: F401

    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    width = s.get_value("width")
    height = s.get_value("height")
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    try:
        # Assumes RGB (3 channels). Copy out so we can unmap immediately.
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
            (height, width, 3)
        ).copy()
    finally:
        buf.unmap(mapinfo)
    # pydxs reads meta by hashing the buffer wrapper (see pydxs README usage).
    buffer_addr = hash(buf)
    return frame, buffer_addr
```

> 보드 검증 시 주의: pydxs README는 probe `info`에 대해 `hash(info)`를 쓰고, appsink 경로에선 buffer 기준 접근이 필요하다. **Task 12(보드)에서 정확한 메타 접근 키(buffer vs probe-info)를 실측 확인**하고 `_gst_sample`/`pydxs_bridge`를 맞춘다.

**Step 4: 통과 확인** → PASS (import 스모크)

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/_gst_sample.py yolo26_4ch_demo/tests/test_gst_sample_import.py
git commit -m "feat: add GstSample->frame/addr extractor (board runtime)"
```

---

## Phase 5: Qt 통합 (배선은 호스트, 표시는 보드)

### Task 8: MainWindow에 dxstream 백엔드 분기 추가

**Files:**
- Modify: `demo/main.py` (`_init_engine_and_workers` 부근, 약 607–730행)

**작업 내용 (단일 책임 분리):**
1. `_init_engine_and_workers`를 `engine_backend` 값으로 분기: `legacy`면 현행 그대로, `dxstream`이면 `_init_dxstream_backend(config)` 호출.
2. `_init_dxstream_backend`:
   - 라벨/팔레트 로드를 위해 모델 클래스명이 필요 → dxstream 모드에서도 클래스명 소스 확보(설계 7.5: 라벨 파일 또는 경량 엔진 메타). 우선 COCO 라벨 상수/파일에서 로드.
   - 채널별 `build_infer_pipeline(...)` 생성, `StreamPipeline(channel_id, ..., on_frame=self._on_native_frame, on_detections=self._on_native_detections)` 생성/`start()`.
3. `_on_native_frame(channel_id, frame)` → `self.frame_ready.emit(channel_id, frame, {"color_format": "rgb"})`.
4. `_on_native_detections(channel_id, det, meta)` → `det = filter_by_classes(det, self.get_selected_classes())` 후 `self.detections_ready.emit(channel_id, det, meta)`.

**TDD 한계:** Qt/gi 전체 기동은 보드. 호스트에서는 `_on_native_frame`/`_on_native_detections`가 올바른 시그널 인자를 만드는지 **순수 함수로 분리**해 테스트한다.

**Step 1: 실패 테스트** — `tests/test_native_signal_wiring.py`

```python
import numpy as np
from demo.native_signal import build_frame_payload, build_detection_payload


def test_build_frame_payload():
    f = np.zeros((4, 4, 3), np.uint8)
    ch, frame, meta = build_frame_payload(1, f)
    assert ch == 1 and meta["color_format"] == "rgb" and frame is f


def test_build_detection_payload_applies_class_filter():
    det = np.array([[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.8, 5]], np.float32)
    ch, out, meta = build_detection_payload(3, det, selected={5})
    assert ch == 3 and out.shape == (1, 6) and out[0, 5] == 5
```

**Step 2: 실패 확인** → FAIL

**Step 3: 최소 구현** — `demo/native_signal.py`

```python
"""Pure helpers that build Qt signal payloads for the native backend.

Separated from main.py so the channel-id/meta/class-filter logic is testable
without Qt or gi.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np

from demo.meta_adapter import filter_by_classes


def build_frame_payload(channel_id: int, frame: np.ndarray) -> Tuple[int, np.ndarray, dict]:
    return channel_id, frame, {"color_format": "rgb"}


def build_detection_payload(
    channel_id: int, detections: np.ndarray, selected: Optional[Set[int]]
) -> Tuple[int, np.ndarray, dict]:
    return channel_id, filter_by_classes(detections, selected), {"color_format": "rgb"}
```

**Step 4: 통과 확인** → PASS

**Step 5: 커밋**

```bash
git add yolo26_4ch_demo/demo/native_signal.py yolo26_4ch_demo/tests/test_native_signal_wiring.py
git commit -m "feat: add pure Qt signal payload builders for native backend"
```

### Task 9: main.py에 백엔드 분기 배선

**Files:**
- Modify: `demo/main.py`

**작업 내용:** Task 8 설계대로 `_init_engine_and_workers` 분기 + `_init_dxstream_backend` + `_on_native_frame`/`_on_native_detections`(내부에서 `build_frame_payload`/`build_detection_payload` 사용) 추가. `import`에 `from demo.native_pipeline import build_infer_pipeline, PreprocessCfg, InferCfg, PostprocessCfg`, `from demo.stream_pipeline import StreamPipeline`, `from demo.native_signal import build_frame_payload, build_detection_payload` 추가. 채널 종료 시 `StreamPipeline.stop()` 호출을 `closeEvent`/정리 경로에 연결.

**검증:** 호스트에서 `python -c "import ast; ast.parse(open('demo/main.py').read())"` 구문 OK + 전체 pytest 통과(legacy 경로 회귀 없음). 실제 표시는 보드.

**Step: 커밋**

```bash
git add yolo26_4ch_demo/demo/main.py
git commit -m "feat: wire dxstream backend branch into MainWindow"
```

---

## Phase 6: 보드 통합 검증 & 마무리 (RK3588)

> 모든 단계는 rockpi에서 `engine_backend: dxstream`로 수행. 각 단계는 Manual verification 게이트.

### Task 10: 단일 채널 네이티브 파이프라인 gst-launch 스모크
- 채널 1개만 `enabled: true`로 두고, 설계의 파이프라인 문자열을 `gst-launch-1.0`로 직접 실행해 `dxosd ! autovideosink`로 검출이 그려지는지 확인(파이프라인 자체 검증).
- Manual verification: 박스가 보이고 FPS가 legacy보다 높은가.

### Task 11: 데모 1채널 dxstream 모드 구동
- `engine_backend: dxstream`, 1채널. Qt 창에 프레임+오버레이 표시되는지.
- Manual verification: 프레임 표시 OK, 박스 위치 정확(원본 좌표 매핑), 크래시 없음.

### Task 12: pydxs 메타 접근 키 실측 보정
- appsink 콜백에서 `DXFrameMeta`를 얻는 정확한 키(buffer 기반 vs probe-info 기반) 확인. 필요 시 `dxpostprocess` 다음에 **pad probe**를 달아 `pydxs.dx_get_frame_meta(hash(info))`로 읽고, frame은 appsink로 받는 2-경로로 보정.
- Manual verification: 검출 개수/라벨이 콘솔 로그와 화면 오버레이에서 일치.

### Task 13: 4채널 확장 + 성능 측정
- 4채널 모두 dxstream. 채널당 FPS / input drop / overall FPS를 legacy와 비교 측정(throughput_stats 재사용).
- Manual verification: rockpi 채널당 FPS가 legacy(~22) 대비 유의미 상승. 목표: NPU 한계 근접 / 호스트 수준.

### Task 14: 클래스 필터·FPS·EOF 루프 통합 점검
- 클래스 필터 체크박스가 dxstream 경로 오버레이에 반영되는지.
- 비디오 EOF 시 루프(파이프라인 재시작 또는 `urisourcebin` looping/seek) 동작.
- max_fps 레이트 리밋이 필요하면 파이프라인에 `videorate`/`identity sleep` 등으로 재현.
- Manual verification: 필터 토글 반영, 무한 재생, drop 거동 정상.

### Task 15: 문서/정리 + legacy 폴백 유지 결정
- README에 dxstream 모드 실행법(venv-dx_stream, GST_PLUGIN_PATH, 모델 경로) 추가.
- 측정 결과를 설계 문서 §7.6에 기록. legacy 제거 여부는 측정 후 별도 결정(우선은 플래그로 양립 유지).
- 최종 전체 pytest 통과 확인.

```bash
git add -A && git commit -m "docs: dxstream backend usage + RK3588 perf results"
```

---

## 완료 기준 (Definition of Done)

- [ ] 호스트 pytest 전부 통과(legacy 회귀 0, 신규 어댑터/빌더/배선 테스트 통과).
- [ ] `engine_backend: legacy` 동작이 기존과 100% 동일(기본값 유지).
- [ ] `engine_backend: dxstream`에서 RK3588 4채널이 표시+오버레이 정상, 채널당 FPS가 legacy 대비 상승.
- [ ] 클래스 필터/FPS 패널/오버레이가 dxstream 경로에서 동작.
- [ ] dxstream 실행법 문서화, 성능 수치 기록.

## 미해결/보드에서 확정할 항목

1. pydxs 메타 접근 키(buffer vs probe-info) — Task 12.
2. appsink 출력 포맷(RGB vs I420) 및 `_gst_sample` 파싱 — Task 11/12.
3. EOF 루프/`max_fps` 레이트 리밋의 파이프라인 재현 방식 — Task 14.
4. `yolo26n.dxnn`(dx_stream 샘플) vs 데모 보유 모델 호환/라벨셋 — Task 10.
5. per-stream vs selector(shared infer) 성능 비교 — Task 13 결과로 결정.
