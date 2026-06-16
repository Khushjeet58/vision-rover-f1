from __future__ import annotations

from http.client import HTTPResponse
import os
import threading
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Optional, Any, Callable

import websocket

from config import RoverConfig
from core.event_bus import SystemEvents, bus
from modules.rover_types import ConnectionState, ConnectionStatus


class VisionStream:
    """Receives the latest JPEG frame from a camera WebSocket or MJPEG HTTP stream."""

    def __init__(self, url: str, config: RoverConfig) -> None:
        self._url = url
        self._config = config
        self._parsed_url = urlparse(url)
        self._transport = self._detect_transport(url)
        self._snapshot_candidates = self._build_snapshot_candidates(url)
        self._lock = threading.Lock()
        self._latest: Optional[Any] = None
        self._connected_since_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._last_message_monotonic = 0.0
        self._source_fps = 0.0
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._app: Optional[websocket.WebSocketApp] = None
        self._http_stream: Optional[HTTPResponse] = None
        self._stale_logged = False
        self._feed_stale = False
        self._had_frames = False
        self._last_error_text = ""
        self._last_error_time = 0.0

    def _detect_transport(self, url: str) -> str:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        path = parsed.path.lower()
        if scheme in {"ws", "wss"}:
            return "websocket"
        if path.endswith((".jpg", ".jpeg")):
            return "jpeg_snapshot"
        return "mjpeg"

    def _build_snapshot_candidates(self, url: str) -> list[str]:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return []

        base = f"{parsed.scheme}://{parsed.netloc}"
        current = parsed.path.lower() or "/"
        candidates: list[str] = []
        for path in ("/cam-hi.jpg", "/cam-lo.jpg", "/capture.jpg", "/jpg"):
            if path != current:
                candidates.append(base + path)
        return candidates

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="VisionStream_Thread")
        self._thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_run,
            daemon=True,
            name="VisionStreamWatchdog_Thread",
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._running = False
        app = self._app
        http_stream = self._http_stream
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        if http_stream is not None:
            try:
                http_stream.close()
            except Exception:
                pass

    def get_latest_frame(self) -> Optional[Any]:
        self._refresh_stale_state()
        with self._lock:
            return self._latest

    def frame_age(self) -> float:
        if not self._last_frame_monotonic:
            return float("inf")
        return time.monotonic() - self._last_frame_monotonic

    def source_fps(self) -> float:
        return self._source_fps

    def is_connected(self) -> bool:
        self._refresh_stale_state()
        return self._connected

    def _set_state(self, state: ConnectionState, detail: str = "") -> None:
        connected = state == ConnectionState.CONNECTED
        if self._connected != connected or detail:
            self._connected = connected
            bus.emit(
                SystemEvents.CONNECTION_STATUS_CHANGED,
                ConnectionStatus(channel="camera", state=state, detail=detail),
            )

    def _on_message(self, _app, message) -> None:
        if isinstance(message, bytes) and message:
            self._ingest_frame(message)

    def _ingest_frame(self, payload: Any) -> None:
        now = time.monotonic()
        if self._last_message_monotonic:
            dt = max(1e-3, now - self._last_message_monotonic)
            instant = 1.0 / dt
            self._source_fps = (0.85 * self._source_fps) + (0.15 * instant) if self._source_fps else instant
        self._last_message_monotonic = now
        with self._lock:
            self._latest = payload
        self._last_frame_monotonic = now
        self._stale_logged = False
        self._feed_stale = False
        self._had_frames = True
        self._set_state(ConnectionState.CONNECTED)

    def _on_open(self, _app) -> None:
        self._on_transport_open("socket open")
        bus.emit(SystemEvents.LOG_MESSAGE, "[VisionStream] Camera socket open. Waiting for first frame.")

    def _on_transport_open(self, detail: str) -> None:
        now = time.monotonic()
        self._connected_since_monotonic = now
        self._last_message_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._source_fps = 0.0
        self._stale_logged = False
        self._feed_stale = False
        self._had_frames = False
        with self._lock:
            self._latest = None
        self._set_state(ConnectionState.CONNECTING, detail)

    def _on_error(self, _app, error) -> None:
        text = str(error)
        self._set_state(ConnectionState.ERROR, text)
        now = time.monotonic()
        if text != self._last_error_text or (now - self._last_error_time) >= 8.0:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] {text}")
            self._last_error_text = text
            self._last_error_time = now

    def _on_close(self, _app, _status_code, _message) -> None:
        self._connected_since_monotonic = 0.0
        self._last_message_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._source_fps = 0.0
        self._feed_stale = False
        had_frames = self._had_frames
        self._had_frames = False
        with self._lock:
            self._latest = None
        self._set_state(ConnectionState.DISCONNECTED)
        if had_frames:
            bus.emit(SystemEvents.LOG_MESSAGE, "[VisionStream] Connection closed.")

    def _run(self) -> None:
        if self._transport == "jpeg_snapshot":
            self._run_jpeg_snapshot(self._url)
            return
        if self._transport == "mjpeg":
            self._run_mjpeg()
            return
        self._run_websocket()

    def _run_websocket(self) -> None:
        self._set_state(ConnectionState.CONNECTING)
        try:
            while self._running:
                try:
                    app = websocket.WebSocketApp(
                        self._url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )
                    with self._lock:
                        self._app = app
                    app.run_forever(ping_interval=0, skip_utf8_validation=True)
                except Exception as exc:
                    self._set_state(ConnectionState.ERROR, str(exc))
                    bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] run error: {exc}")

                if self._running:
                    self._set_state(ConnectionState.CONNECTING)
                    time.sleep(self._config.reconnect_interval)
        finally:
            self._set_state(ConnectionState.DISCONNECTED)

    def _run_mjpeg(self) -> None:
        # Mirror the known-good capture path used in the standalone ESP32 test app.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"
        try:
            import cv2  # type: ignore
        except Exception as exc:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] OpenCV unavailable ({exc}). Starting in raw HTTP mode.")
            cv2 = None

        self._set_state(ConnectionState.CONNECTING)
        transports: list[tuple[str, Callable[[], bool]]] = [("raw-http", self._run_mjpeg_raw_http_once)]
        if self._snapshot_candidates:
            transports.append(("snapshot", self._run_snapshot_fallbacks_once))
        if cv2 is not None and bool(getattr(self._config, "camera_enable_ffmpeg_fallback", False)):
            transports.append(("ffmpeg", lambda: self._run_mjpeg_ffmpeg_once(cv2)))

        try:
            while self._running:
                cycle_had_frames = False
                for name, runner in transports:
                    if not self._running:
                        break
                    if name != "ffmpeg":
                        bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] Trying {name} camera transport.")
                    had_frames = runner()
                    cycle_had_frames = cycle_had_frames or had_frames
                    if had_frames:
                        break
                if not self._running:
                    break
                self._set_state(ConnectionState.CONNECTING)
                time.sleep(0.5 if cycle_had_frames else self._config.reconnect_interval)
        finally:
            self._set_state(ConnectionState.DISCONNECTED)

    def _run_mjpeg_ffmpeg_once(self, cv2) -> bool:
        cap = None
        had_frames = False
        try:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise ConnectionError("cannot open camera stream")

            self._on_transport_open("stream open")
            bus.emit(SystemEvents.LOG_MESSAGE, "[VisionStream] MJPEG capture open. Waiting for first frame.")

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ConnectionError("camera frame read failed")
                had_frames = True
                self._ingest_frame(frame)
        except Exception as exc:
            self._on_error(None, exc)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._on_close(None, None, None)
        return had_frames

    def _run_mjpeg_raw_http_once(self) -> bool:
        self._set_state(ConnectionState.CONNECTING)
        headers = {
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "User-Agent": "VISION/1.0",
            "Accept": "multipart/x-mixed-replace,image/jpeg,*/*",
        }
        read_size = max(2048, int(getattr(self._config, "camera_http_read_size", 16_384)))
        timeout = max(1.0, self._config.ws_recv_timeout)
        had_frames = False

        stream: Optional[HTTPResponse] = None
        try:
            req = Request(self._url, headers=headers)
            stream = urlopen(req, timeout=timeout)
            with self._lock:
                self._http_stream = stream
            self._on_transport_open("stream open")
            bus.emit(
                SystemEvents.LOG_MESSAGE,
                "[VisionStream] Low-latency MJPEG stream open via urllib. Waiting for first frame.",
            )

            buffer = bytearray()
            while self._running:
                chunk = self._read_http_chunk(stream, read_size)
                if not chunk:
                    raise ConnectionError("camera stream ended")
                self._consume_mjpeg_chunk(chunk, buffer)
                had_frames = had_frames or self._had_frames
        except URLError as exc:
            self._on_error(None, exc.reason if getattr(exc, "reason", None) else exc)
        except Exception as exc:
            self._on_error(None, exc)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            with self._lock:
                self._http_stream = None
            self._on_close(None, None, None)
        return had_frames

    @staticmethod
    def _read_http_chunk(stream: HTTPResponse, read_size: int) -> bytes:
        reader = getattr(stream, "read1", None)
        if callable(reader):
            return reader(read_size)
        fp = getattr(stream, "fp", None)
        reader = getattr(fp, "read1", None)
        if callable(reader):
            return reader(read_size)
        return stream.read(read_size)

    def _run_snapshot_fallbacks_once(self) -> bool:
        for candidate in self._snapshot_candidates:
            if not self._running:
                return False
            bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] Trying snapshot fallback: {candidate}")
            if self._run_jpeg_snapshot(candidate, trial_seconds=4.0):
                return True
        return False

    def _run_jpeg_snapshot(self, url: str, trial_seconds: float | None = None) -> bool:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            self._on_error(None, f"snapshot mode unavailable: {exc}")
            return False

        headers = {
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
            "User-Agent": "VISION/1.0",
            "Accept": "image/jpeg,*/*",
        }
        timeout = max(1.0, self._config.ws_recv_timeout)
        min_period = 1.0 / max(1, self._config.snapshot_poll_hz)
        had_frames = False
        started = time.monotonic()
        self._set_state(ConnectionState.CONNECTING)

        while self._running:
            loop_started = time.monotonic()
            if trial_seconds is not None and had_frames is False and (loop_started - started) >= trial_seconds:
                return False
            try:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=timeout) as response:
                    payload = response.read()
                if not payload:
                    raise ConnectionError("empty JPEG snapshot")
                arr = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ConnectionError("snapshot decode failed")
                if not had_frames:
                    self._on_transport_open("snapshot open")
                    bus.emit(SystemEvents.LOG_MESSAGE, f"[VisionStream] Snapshot stream active: {url}")
                self._ingest_frame(frame)
                had_frames = True
            except Exception as exc:
                self._on_error(None, exc)
                if had_frames:
                    self._on_close(None, None, None)
                    return False
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, min_period - elapsed))
        return had_frames

    def _consume_mjpeg_chunk(self, chunk: bytes, buffer: bytearray) -> None:
        if not chunk:
            return

        buffer.extend(chunk)
        latest_payload: bytes | None = None
        while True:
            soi = buffer.find(b"\xff\xd8")
            if soi < 0:
                if len(buffer) > 256_000:
                    del buffer[:-2]
                break
            if soi > 0:
                del buffer[:soi]

            eoi = buffer.find(b"\xff\xd9", 2)
            if eoi < 0:
                if len(buffer) > 1_000_000:
                    del buffer[: len(buffer) - 256_000]
                break

            # Keep only the newest complete JPEG from this network read. If the
            # app is busy, older frames are discarded instead of becoming latency.
            latest_payload = bytes(buffer[: eoi + 2])
            del buffer[: eoi + 2]

        if latest_payload is not None:
            self._ingest_frame(latest_payload)

    def _watchdog_run(self) -> None:
        interval = min(0.25, max(0.05, self._config.camera_disconnect_timeout / 4))
        while self._running:
            self._refresh_stale_state()
            time.sleep(interval)

    def _refresh_stale_state(self) -> None:
        if not self._running:
            return

        now = time.monotonic()
        connected_since = self._connected_since_monotonic
        last_frame = self._last_frame_monotonic
        if not connected_since:
            return

        if last_frame:
            age = now - last_frame
            if age <= self._config.frame_stale_seconds:
                return
            if not self._feed_stale:
                self._feed_stale = True
                self._source_fps = 0.0
                with self._lock:
                    self._latest = None
                self._set_state(ConnectionState.DISCONNECTED, "stale feed")
                if not self._stale_logged:
                    self._stale_logged = True
                    bus.emit(
                        SystemEvents.LOG_MESSAGE,
                        "[VisionStream] Camera feed is stale. Clearing frozen frame and waiting for recovery.",
                    )
            return

        if (now - connected_since) <= self._config.camera_initial_frame_timeout:
            return
        if not self._feed_stale:
            self._feed_stale = True
            self._source_fps = 0.0
            with self._lock:
                self._latest = None
            self._set_state(ConnectionState.DISCONNECTED, "no frames yet")
            if not self._stale_logged:
                self._stale_logged = True
                bus.emit(
                    SystemEvents.LOG_MESSAGE,
                    "[VisionStream] Camera socket is open but no frame has arrived yet. Waiting without reconnecting.",
                )
