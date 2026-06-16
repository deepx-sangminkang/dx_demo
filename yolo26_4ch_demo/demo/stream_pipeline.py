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
import time
from collections import OrderedDict
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
        meta_src_name: Optional[str] = None,
        loop: bool = False,
        source_size_callback: Optional[Callable[[int, int, int], None]] = None,
        pace_fps: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.channel_id = channel_id
        self.pipeline_str = pipeline_str
        self.bridge = bridge
        self.frame_callback = frame_callback
        self.detection_callback = detection_callback
        self.appsink_name = appsink_name
        self.error_callback = error_callback
        self.meta_src_name = meta_src_name
        self.loop = loop
        # Called once (channel_id, width, height) with the original detection
        # coordinate-space resolution, read from the meta (dxpostprocess) pad caps
        # so the overlay can map boxes correctly even when the displayed frame is
        # downscaled by a display-branch dxscale.
        self.source_size_callback = source_size_callback
        self._source_size_reported = False
        self._segment_armed = False
        # Native-fps pacing: the appsink runs sync=false (so gapless SEGMENT-loop
        # seeks are not stalled by GstBaseSink clock-sync) and instead this class
        # paces delivery to the source PTS in the streaming thread. The sleep
        # creates backpressure that throttles the whole pipeline to the video's
        # real frame rate, so the NPU/VPU do not run flat out. The pacing anchor
        # is reset on a PTS discontinuity (the loop seam, where PTS jumps back to
        # 0) so looping stays seamless instead of sleeping a whole clip-length.
        self.pace_fps = pace_fps
        self._monotonic = monotonic
        self._sleep = sleep
        self._pace_anchor_wall: Optional[float] = None
        self._pace_anchor_pts: Optional[int] = None
        self._pace_last_pts: Optional[int] = None
        # Cap a single pacing sleep so a bad/huge PTS gap can never freeze a tile.
        self._PACE_MAX_SLEEP = 0.5

        self._gst = gst
        self._extract = sample_extractor
        self._sample_count = 0
        # Detections captured on the meta-source pad (before videoconvert),
        # keyed by buffer PTS, consumed by the appsink frame with the same PTS.
        self._meta_stash: "OrderedDict[int, object]" = OrderedDict()
        self._meta_stash_lock = threading.Lock()
        self._META_STASH_MAX = 240
        # Last detections successfully resolved for a frame, reused for up to
        # _MAX_META_REUSE consecutive stash misses so a single loop-boundary
        # frame (whose PTS desyncs from the stash right after a seek) does not
        # flash an empty overlay.
        self._last_detections = None
        self._meta_miss_streak = 0
        self._meta_reused = False
        self._MAX_META_REUSE = 2
        # Stall watchdog: RK3588 loop seeks occasionally wedge a single channel
        # under VPU contention (one tile freezes while the others keep playing).
        # If a looping channel produces no frames for _STALL_TIMEOUT seconds, a
        # flushing seek is issued to kick it back into motion. The timeout is
        # kept short (and the watchdog polled every second) so a wedged loop is
        # recovered in ~2s instead of producing a long, visible freeze -- at
        # ~95 fps a genuinely-playing channel never has a 2s frame gap, so this
        # only ever fires on a real wedge.
        self._last_sample_mono: Optional[float] = None
        self._stall_recoveries = 0
        self._STALL_TIMEOUT = 2.0
        self._WATCHDOG_INTERVAL = 1
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

        # Log which decoder ``decodebin`` actually instantiates so a silent
        # fallback to software decoding (e.g. avdec_h264 instead of the HW
        # mppvideodec) is visible instead of just showing high CPU usage.
        try:
            self._pipeline.connect("deep-element-added", self._on_deep_element_added)
        except Exception as exc:  # pragma: no cover - board glue
            logger.debug("deep-element-added connect failed (ch %s): %s",
                         self.channel_id, exc)

        # Capture detection metadata on the meta-source src pad (before the
        # NV12->RGB videoconvert, which drops the custom DXFrameMeta). The probe
        # stashes detections by buffer PTS for the appsink frame to pick up.
        if self.meta_src_name:
            meta_el = self._pipeline.get_by_name(self.meta_src_name)
            if meta_el is not None:
                src_pad = meta_el.get_static_pad("src")
                if src_pad is not None:
                    src_pad.add_probe(
                        self._gst.PadProbeType.BUFFER, self._on_meta_probe
                    )
                else:
                    print(
                        f"[WARN] Channel {self.channel_id}: meta element "
                        f"'{self.meta_src_name}' has no src pad; detections "
                        f"disabled",
                        file=sys.stderr, flush=True,
                    )
            else:
                print(
                    f"[WARN] Channel {self.channel_id}: meta element "
                    f"'{self.meta_src_name}' not found; detections disabled",
                    file=sys.stderr, flush=True,
                )

        # Watch the pipeline bus so element errors/warnings (e.g. dxinfer failing
        # to load the model, dxpostprocess missing its library, caps negotiation
        # failures) surface instead of silently yielding zero frames.
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)
        bus.connect("message::eos", self._on_bus_eos)
        bus.connect("message::segment-done", self._on_bus_segment_done)
        bus.connect("message::async-done", self._on_bus_async_done)

        # Go straight to PLAYING without a synchronous PAUSED-preroll wait. The
        # four channels share the RK3588 decoder, so blocking on preroll here
        # deadlocks the sequential startup (a later channel never prerolls while
        # earlier ones already hold the VPU). SEGMENT looping is armed later from
        # the async-done handler, once this pipeline has actually prerolled --
        # arming a non-flushing seek before preroll completes would hang.
        self._loop = GLib.MainLoop()
        self._pipeline.set_state(self._gst.State.PLAYING)
        if self._should_loop():
            # Periodic watchdog: recover a channel whose loop seek wedged the
            # decoder (frozen tile) by issuing a flushing seek.
            GLib.timeout_add_seconds(self._WATCHDOG_INTERVAL, self._check_stall)
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

    def _on_deep_element_added(self, bin_, sub_bin, element):  # pragma: no cover - board glue
        """Log the video decoder ``decodebin`` selected (HW vs SW)."""
        try:
            factory = element.get_factory()
            klass = factory.get_klass() if factory is not None else ""
            if "Decoder/Video" in (klass or ""):
                name = factory.get_name() if factory is not None else "?"
                hw = "HW" if "mpp" in name.lower() else "SW"
                print(
                    f"[INFO] Channel {self.channel_id}: video decoder = {name} "
                    f"({hw})",
                    flush=True,
                )
        except Exception as exc:
            logger.debug("decoder-log failed (ch %s): %s", self.channel_id, exc)

    def _on_new_sample_signal(self, appsink):  # pragma: no cover - board glue
        sample = appsink.emit("pull-sample")
        return self._on_new_sample(sample)

    def _on_new_sample(self, appsink_with_sample):
        """Dispatch one appsink sample. Never raises into GStreamer."""
        try:
            frame, gst_buffer = self._extract(appsink_with_sample)
            if frame is None:
                return self._gst.FlowReturn.OK

            if self.pace_fps:
                self._pace(gst_buffer)
            self._note_sample_time()
            detections = self._detections_for_sample(gst_buffer)
            self._log_detection_diagnostics(detections)
            self.frame_callback(self.channel_id, frame)
            self.detection_callback(self.channel_id, detections)
        except Exception as exc:
            logger.exception("stream sample dispatch failed (ch %s): %s",
                             self.channel_id, exc)
        return self._gst.FlowReturn.OK

    def _pace(self, gst_buffer) -> None:
        """Throttle the streaming thread so frames are delivered at source fps.

        Sleeps until wall-clock time matches the buffer PTS (relative to a moving
        anchor). The anchor is (re)set on the first frame and whenever PTS jumps
        backwards -- the gapless loop seam -- so wrapping from end-of-clip back to
        the start never incurs a clip-length sleep. The sleep itself backpressures
        the upstream decode/NPU, capping the whole pipeline to real-time fps.
        """

        pts = self._buffer_pts(gst_buffer)
        if pts is None:
            return
        now = self._monotonic()
        if (self._pace_anchor_wall is None
                or self._pace_last_pts is None
                or pts < self._pace_last_pts):
            self._pace_anchor_wall = now
            self._pace_anchor_pts = pts
            self._pace_last_pts = pts
            return
        self._pace_last_pts = pts
        target = self._pace_anchor_wall + (pts - self._pace_anchor_pts) / 1e9
        delay = target - self._monotonic()
        if delay > 0:
            self._sleep(min(delay, self._PACE_MAX_SLEEP))

    def _detections_for_sample(self, gst_buffer):
        """Resolve detections for an appsink buffer.

        Prefer detections captured on the meta-source pad (matched by PTS, since
        the custom DXFrameMeta does not survive the NV12->RGB videoconvert that
        produces the appsink buffer). Fall back to reading meta straight off the
        appsink buffer when no probe is active (e.g. host tests / passthrough).
        """

        pts = self._buffer_pts(gst_buffer)
        if pts is not None:
            stashed = self._take_meta(pts)
            if stashed is not None:
                self._last_detections = stashed
                self._meta_miss_streak = 0
                self._meta_reused = False
                return stashed
        direct = self.bridge.detections_for_buffer(gst_buffer)
        if getattr(self.bridge, "last_meta_present", False):
            self._last_detections = direct
            self._meta_miss_streak = 0
            self._meta_reused = False
            return direct
        # No metadata for this frame from either source. Right after a loop seek
        # the PTS timeline restarts and one appsink frame can briefly desync from
        # the stash; reuse the last known detections for a bounded number of
        # frames so the overlay stays stable instead of flashing empty.
        if (self._last_detections is not None
                and self._meta_miss_streak < self._MAX_META_REUSE):
            self._meta_miss_streak += 1
            self._meta_reused = True
            return self._last_detections
        self._meta_miss_streak += 1
        self._meta_reused = False
        return direct

    # ----- PTS-correlated metadata stash (host-testable) -----

    def _buffer_pts(self, gst_buffer) -> Optional[int]:
        """Return a usable PTS for a buffer, or None when invalid/unavailable."""

        pts = getattr(gst_buffer, "pts", None)
        if pts is None:
            return None
        gst = self._gst
        invalid = getattr(gst, "CLOCK_TIME_NONE", None) if gst is not None else None
        if invalid is not None and pts == invalid:
            return None
        return pts

    def _on_meta_probe(self, pad, info, *_):  # pragma: no cover - board glue
        """Pad probe: read DXFrameMeta before videoconvert, stash by PTS."""
        try:
            buf = info.get_buffer()
            if buf is None:
                return self._gst.PadProbeReturn.OK
            self._maybe_report_source_size(pad)
            detections = self.bridge.detections_for_buffer(buf)
            pts = self._buffer_pts(buf)
            if pts is not None:
                self._stash_meta(pts, detections)
        except Exception as exc:
            logger.exception("meta probe failed (ch %s): %s", self.channel_id, exc)
        return self._gst.PadProbeReturn.OK

    def _maybe_report_source_size(self, pad):  # pragma: no cover - board glue
        """Report the original (pre-downscale) frame size once, from pad caps."""
        if self._source_size_reported or self.source_size_callback is None:
            return
        try:
            caps = pad.get_current_caps()
            if caps is None or caps.get_size() == 0:
                return
            structure = caps.get_structure(0)
            ok_w, width = structure.get_int("width")
            ok_h, height = structure.get_int("height")
            if ok_w and ok_h and width > 0 and height > 0:
                self._source_size_reported = True
                self.source_size_callback(self.channel_id, int(width), int(height))
        except Exception as exc:
            logger.debug("source-size probe failed (ch %s): %s", self.channel_id, exc)

    def _stash_meta(self, pts: int, detections) -> None:
        """Store detections for a PTS, bounding total memory used."""
        with self._meta_stash_lock:
            self._meta_stash[pts] = detections
            self._meta_stash.move_to_end(pts)
            while len(self._meta_stash) > self._META_STASH_MAX:
                self._meta_stash.popitem(last=False)

    def _clear_meta_stash(self) -> None:
        """Drop all stashed metadata (used when a seek restarts the timeline)."""
        with self._meta_stash_lock:
            self._meta_stash.clear()

    def _take_meta(self, pts: int):
        """Pop detections for a PTS (and drop any older, now-stale entries)."""
        with self._meta_stash_lock:
            if pts not in self._meta_stash:
                return None
            # Buffers are produced in order, so anything inserted before this PTS
            # belongs to frames the appsink already dropped/consumed.
            while True:
                oldest = next(iter(self._meta_stash))
                value = self._meta_stash.pop(oldest)
                if oldest == pts:
                    return value

    # ----- detection diagnostics (host-testable) -----

    #: Log the detection state for this many initial samples per channel, then
    #: only every Nth sample, so a blank-overlay run reveals whether the issue is
    #: missing metadata vs. zero detected objects vs. an overlay problem.
    _DEBUG_FIRST = 3
    _DEBUG_EVERY = 300

    def _should_log_sample(self, count: int) -> bool:
        if count <= self._DEBUG_FIRST:
            return True
        return count % self._DEBUG_EVERY == 0

    def _log_detection_diagnostics(self, detections) -> None:
        """Print metadata/detection state for diagnostic samples."""

        self._sample_count += 1
        if not self._should_log_sample(self._sample_count):
            return
        meta_present = getattr(self.bridge, "last_meta_present", None)
        n = int(detections.shape[0]) if hasattr(detections, "shape") else len(detections)
        # When meta was missing for this frame we reuse the previous frame's
        # detections (loop-seam glitch); flag it so the False is not mistaken for
        # a real, visible metadata dropout.
        note = " (reused prev boxes)" if self._meta_reused else ""
        print(
            f"[DXS-DEBUG] ch{self.channel_id} sample#{self._sample_count}: "
            f"frame_meta_present={meta_present} detections={n}{note}",
            file=sys.stderr,
            flush=True,
        )

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
        # With SEGMENT looping the stream never reaches EOS; if we still get one
        # (e.g. SEGMENT seek unsupported by an element), fall back to a deferred
        # flush-seek so playback at least attempts to restart.
        if self._should_loop():
            from gi.repository import GLib  # type: ignore

            GLib.idle_add(self._deferred_segment_restart)
            msg = f"Channel {self.channel_id}: end of stream -> looping"
            logger.info(msg)
            print(f"[INFO] {msg}", file=sys.stderr, flush=True)
            return
        msg = f"Channel {self.channel_id}: end of stream (EOS)"
        logger.info(msg)
        print(f"[INFO] {msg}", file=sys.stderr, flush=True)

    def _on_bus_segment_done(self, _bus, _message):  # pragma: no cover - board glue
        # Reaching the end of the SEGMENT: immediately re-arm a non-flushing
        # SEGMENT seek back to the start for gapless looping.
        try:
            self._segment_seek(flush=False)
        except Exception as exc:
            logger.exception("segment loop seek failed (ch %s): %s",
                             self.channel_id, exc)

    def _on_bus_async_done(self, _bus, _message):  # pragma: no cover - board glue
        # The pipeline has prerolled (reached PLAYING). Arm the SEGMENT loop now,
        # exactly once, with a non-flushing seek: doing it post-preroll avoids
        # both the startup deadlock (no synchronous PAUSED wait) and the
        # 'Got data flow before segment event' warnings (a non-flush seek on an
        # already-running pipeline keeps the live segment valid).
        if self._should_arm_segment_loop():
            self._segment_armed = True
            try:
                self._segment_seek(flush=False)
            except Exception as exc:
                logger.exception("segment loop arm failed (ch %s): %s",
                                 self.channel_id, exc)

    def _should_arm_segment_loop(self) -> bool:
        """Whether the SEGMENT loop still needs arming on this pipeline."""

        return self._should_loop() and not self._segment_armed

    def _deferred_segment_restart(self):  # pragma: no cover - board glue
        try:
            self._segment_seek(flush=True)
        except Exception as exc:
            logger.exception("loop restart failed (ch %s): %s",
                             self.channel_id, exc)
        return False  # one-shot idle source

    def _should_loop(self) -> bool:
        """Whether this source should rewind to play again instead of stopping."""

        return self.loop and self._pipeline is not None

    # ----- stall watchdog (host-testable) -----

    def _note_sample_time(self) -> None:
        """Record that a frame was just delivered (for the stall watchdog)."""

        self._last_sample_mono = time.monotonic()

    def _is_stalled(self, now: float) -> bool:
        """True when a looping channel has produced no frames for too long.

        Returns False before the first frame (startup is allowed to be slow
        under VPU contention) and for non-looping channels (a finite clip that
        legitimately ends must not be 'recovered').
        """

        if not self._should_loop():
            return False
        last = self._last_sample_mono
        if last is None:
            return False
        return (now - last) > self._STALL_TIMEOUT

    def _check_stall(self):  # pragma: no cover - board glue
        try:
            if self._is_stalled(time.monotonic()):
                self._stall_recoveries += 1
                msg = (f"Channel {self.channel_id}: no frames for "
                       f"{self._STALL_TIMEOUT:.0f}s -> recovering loop")
                logger.warning(msg)
                print(f"[WARN] {msg}", file=sys.stderr, flush=True)
                # A flushing seek resets the wedged decoder; mark the loop armed
                # so the normal SEGMENT_DONE handler keeps it going afterwards.
                self._segment_armed = True
                self._segment_seek(flush=True)
                self._note_sample_time()  # avoid immediately re-firing
        except Exception as exc:
            logger.exception("stall watchdog failed (ch %s): %s",
                             self.channel_id, exc)
        return True  # keep the periodic source alive

    def _segment_seek(self, flush: bool) -> None:
        """Seek to the start using a SEGMENT seek for gapless looping.

        ``flush=True`` is used to arm the segment loop (and to recover from a
        stray EOS); the continuation seek issued on SEGMENT_DONE is non-flushing
        so playback wraps around without a visible gap.
        """

        gst = self._gst
        flags = gst.SeekFlags.SEGMENT
        if flush:
            flags |= gst.SeekFlags.FLUSH
        # The seek restarts the PTS timeline; drop any stashed metadata so a new
        # frame can never match stale detections left from the previous loop.
        self._clear_meta_stash()
        self._pipeline.seek(
            1.0,
            gst.Format.TIME,
            flags,
            gst.SeekType.SET,
            0,
            gst.SeekType.SET,
            -1,
        )
