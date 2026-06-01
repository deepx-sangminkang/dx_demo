"""Run a native dx_stream inference pipeline and bridge samples to Qt callbacks.

One ``StreamPipeline`` per channel: it parses a launch string ending in an
``appsink``, and on every new sample pulls the decoded frame + GStreamer buffer,
reads detections from the buffer via :class:`~demo.pydxs_bridge.PydxsBridge`,
and dispatches both to the supplied callbacks (which marshal onto the Qt thread).

The GStreamer runtime (``gi``) and frame extraction are board-only and injected,
so the sample-dispatch logic is unit-testable on the dev host without HW.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FrameCallback = Callable[[int, np.ndarray], None]
DetectionCallback = Callable[[int, np.ndarray], None]
ErrorCallback = Callable[[int, str], None]
SampleExtractor = Callable[[object], Tuple[Optional[np.ndarray], Optional[object]]]


def _import_gst():
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore

    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


class StreamPipeline:
    """A single-channel native inference pipeline feeding Qt callbacks."""

    def __init__(
        self,
        channel_id: int,
        pipeline_str: str,
        bridge,
        frame_callback: FrameCallback,
        detection_callback: DetectionCallback,
        appsink_name: str = "sink",
        gst=None,
        sample_extractor: Optional[SampleExtractor] = None,
        error_callback: Optional[ErrorCallback] = None,
    ):
        self.channel_id = channel_id
        self.pipeline_str = pipeline_str
        self.bridge = bridge
        self.frame_callback = frame_callback
        self.detection_callback = detection_callback
        self.appsink_name = appsink_name
        self.error_callback = error_callback

        self._gst = gst
        self._extract = sample_extractor
        self._pipeline = None
        self._loop = None
        self._thread: Optional[threading.Thread] = None

    # ----- lifecycle (board) -----

    def start(self) -> None:  # pragma: no cover - requires GStreamer runtime
        if self._gst is None:
            self._gst = _import_gst()
        if self._extract is None:
            from ._gst_sample import extract_frame_and_buffer

            self._extract = extract_frame_and_buffer

        from gi.repository import GLib  # type: ignore

        self._pipeline = self._gst.parse_launch(self.pipeline_str)
        appsink = self._pipeline.get_by_name(self.appsink_name)
        appsink.connect("new-sample", self._on_new_sample_signal)

        # Watch the pipeline bus so element errors/warnings (e.g. dxinfer failing
        # to load the model, dxpostprocess missing its library, caps negotiation
        # failures) surface instead of silently yielding zero frames.
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)
        bus.connect("message::eos", self._on_bus_eos)

        self._loop = GLib.MainLoop()
        self._pipeline.set_state(self._gst.State.PLAYING)
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - requires GStreamer runtime
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ----- sample handling (host-testable) -----

    def _on_new_sample_signal(self, appsink):  # pragma: no cover - board glue
        sample = appsink.emit("pull-sample")
        return self._on_new_sample(sample)

    def _on_new_sample(self, appsink_with_sample):
        """Dispatch one appsink sample. Never raises into GStreamer."""
        try:
            frame, gst_buffer = self._extract(appsink_with_sample)
            if frame is None:
                return self._gst.FlowReturn.OK

            detections = self.bridge.detections_for_buffer(gst_buffer)
            self.frame_callback(self.channel_id, frame)
            self.detection_callback(self.channel_id, detections)
        except Exception as exc:
            logger.exception("stream sample dispatch failed (ch %s): %s",
                             self.channel_id, exc)
        return self._gst.FlowReturn.OK

    # ----- bus diagnostics (host-testable formatting + dispatch) -----

    def _format_bus_error(self, src_name: str, message: str,
                          debug: Optional[str]) -> str:
        """Build a human-readable error line for a bus error/warning."""

        text = f"Channel {self.channel_id}: GStreamer error from {src_name}: {message}"
        if debug:
            text += f" | {debug}"
        return text

    def _dispatch_bus_error(self, src_name: str, message: str,
                            debug: Optional[str]) -> None:
        """Log a bus error and forward it to the optional error callback.

        Never raises: bus callbacks run on the GLib loop thread.
        """

        text = self._format_bus_error(src_name, message, debug)
        logger.error(text)
        print(f"[ERROR] {text}", file=sys.stderr, flush=True)
        if self.error_callback is not None:
            try:
                self.error_callback(self.channel_id, message)
            except Exception:  # pragma: no cover - defensive
                logger.exception("error_callback raised (ch %s)", self.channel_id)

    def _on_bus_error(self, _bus, message):  # pragma: no cover - board glue
        err, debug = message.parse_error()
        src = message.src.get_name() if message.src is not None else "?"
        self._dispatch_bus_error(src, err.message, debug)

    def _on_bus_warning(self, _bus, message):  # pragma: no cover - board glue
        warn, debug = message.parse_warning()
        src = message.src.get_name() if message.src is not None else "?"
        text = self._format_bus_error(src, warn.message, debug)
        logger.warning(text)
        print(f"[WARN] {text}", file=sys.stderr, flush=True)

    def _on_bus_eos(self, _bus, _message):  # pragma: no cover - board glue
        # File sources reach EOS at end of clip; log so a blank display has a
        # visible explanation. Looping is a future enhancement.
        msg = f"Channel {self.channel_id}: end of stream (EOS)"
        logger.info(msg)
        print(f"[INFO] {msg}", file=sys.stderr, flush=True)
