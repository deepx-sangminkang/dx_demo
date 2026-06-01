"""Cross-channel HW-decode verdict caching in CaptureThread._open_capture.

The HW GStreamer probe is slow and, on boards where it never yields frames,
every channel paying that cost serially can let the window close before the
later channels start. Only the first channel should probe HW; once HW is known
bad the others must skip straight to software.
"""

from __future__ import annotations

import queue

import pytest

from demo import workers


@pytest.fixture(autouse=True)
def _reset_hw_verdict():
    workers._hw_probe_started = False
    workers._hw_known_bad = False
    workers._hw_decode_env = {
        "platform": "rk3588",
        "opencv_gstreamer": True,
        "rga_convert": True,
    }
    yield
    workers._hw_probe_started = False
    workers._hw_known_bad = False
    workers._hw_decode_env = None


def _make_thread(channel_id: int):
    return workers.CaptureThread(
        channel_id=channel_id,
        source=f"/data/{channel_id}.mp4",
        input_queue=queue.Queue(),
        decode_mode="auto",
    )


def test_second_channel_skips_hw_after_first_probe_fails(monkeypatch):
    calls = []

    def fake_open_capture(**kwargs):
        calls.append(kwargs["decode_mode"])
        # HW decode never yields frames on this board: always falls back to SW.
        return object(), False, "fell back to SW", "bgr"

    monkeypatch.setattr(workers.gst, "open_capture", fake_open_capture)

    _make_thread(0)._open_capture()
    _make_thread(1)._open_capture()
    _make_thread(2)._open_capture()

    # Only the first channel probes HW ("auto"); the rest skip to "sw".
    assert calls[0] == "auto"
    assert calls[1] == "sw"
    assert calls[2] == "sw"
    assert workers._hw_known_bad is True


def test_concurrent_second_channel_does_not_wait_for_inflight_probe(monkeypatch):
    """If a probe is already in flight (started but no verdict yet), other
    channels must not block behind it -- they open in software immediately."""

    calls = []

    def fake_open_capture(**kwargs):
        calls.append(kwargs["decode_mode"])
        return object(), False, "fell back to SW", "bgr"

    monkeypatch.setattr(workers.gst, "open_capture", fake_open_capture)

    # Simulate channel 0 having started (but not finished) its HW probe.
    workers._hw_probe_started = True

    _make_thread(1)._open_capture()

    assert calls == ["sw"]


def test_first_channel_still_probes_hw(monkeypatch):
    calls = []

    def fake_open_capture(**kwargs):
        calls.append(kwargs["decode_mode"])
        return object(), True, "HW decode", "rgb"

    monkeypatch.setattr(workers.gst, "open_capture", fake_open_capture)

    t = _make_thread(0)
    t._open_capture()

    assert calls == ["auto"]
    assert t.used_hw is True
    # HW worked, so the verdict stays good (not marked bad).
    assert workers._hw_known_bad is False
