import time

from config import RoverConfig
from modules.vision_stream import VisionStream


def test_on_message_keeps_latest_frame():
    stream = VisionStream("ws://camera", RoverConfig("ws://cam", "ws://servo", "ws://motor"))

    stream._on_message(None, b"frame-1")
    stream._on_message(None, b"frame-2")

    assert stream.get_latest_frame() == b"frame-2"


def test_frame_age_is_finite_after_first_frame():
    stream = VisionStream("ws://camera", RoverConfig("ws://cam", "ws://servo", "ws://motor"))

    stream._on_message(None, b"frame")
    age = stream.frame_age()

    assert age >= 0
    assert age < 1


def test_stale_stream_clears_frame_and_marks_disconnected():
    cfg = RoverConfig(
        "ws://cam",
        "ws://servo",
        "ws://motor",
        frame_stale_seconds=0.01,
        camera_reconnect_timeout=5.0,
    )
    stream = VisionStream("ws://camera", cfg)
    stream._running = True
    stream._connected = True
    stream._connected_since_monotonic = time.monotonic() - 1
    stream._last_frame_monotonic = time.monotonic() - 1
    stream._latest = b"stale"
    closed = {"value": False}
    stream._app = type("DummyApp", (), {"close": lambda self: closed.__setitem__("value", True)})()

    stream._refresh_stale_state()

    assert stream.get_latest_frame() is None
    assert stream.is_connected() is False
    assert stream._source_fps == 0.0
    assert closed["value"] is False


def test_socket_open_waits_for_initial_frame_before_marking_connected():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", camera_initial_frame_timeout=5.0)
    stream = VisionStream("ws://camera", cfg)

    stream._on_open(None)

    assert stream.is_connected() is False
    assert stream._connected_since_monotonic > 0


def test_missing_initial_frame_triggers_reconnect_after_timeout():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", camera_initial_frame_timeout=0.01)
    stream = VisionStream("ws://camera", cfg)
    stream._running = True
    stream._on_open(None)
    stream._connected_since_monotonic = time.monotonic() - 1
    closed = {"value": False}
    stream._app = type("DummyApp", (), {"close": lambda self: closed.__setitem__("value", True)})()

    stream._refresh_stale_state()

    assert stream.get_latest_frame() is None
    assert stream.is_connected() is False
    assert closed["value"] is False


def test_stale_stream_reconnects_after_long_timeout():
    cfg = RoverConfig(
        "ws://cam",
        "ws://servo",
        "ws://motor",
        frame_stale_seconds=0.01,
        camera_reconnect_timeout=0.02,
    )
    stream = VisionStream("ws://camera", cfg)
    stream._running = True
    stream._connected = True
    stream._connected_since_monotonic = time.monotonic() - 1
    stream._last_frame_monotonic = time.monotonic() - 1
    stream._latest = b"stale"
    closed = {"value": False}
    stream._app = type("DummyApp", (), {"close": lambda self: closed.__setitem__("value", True)})()

    stream._refresh_stale_state()

    assert closed["value"] is False


def test_mjpeg_parser_handles_split_frame_chunks():
    cfg = RoverConfig("http://cam/stream", "ws://servo", "ws://motor")
    stream = VisionStream("http://cam/stream", cfg)
    buffer = bytearray()
    jpeg = b"\xff\xd8abc\xff\xd9"

    stream._consume_mjpeg_chunk(jpeg[:4], buffer)
    assert stream.get_latest_frame() is None

    stream._consume_mjpeg_chunk(jpeg[4:], buffer)
    assert stream.get_latest_frame() == jpeg


def test_mjpeg_parser_ignores_noise_and_keeps_latest_frame():
    cfg = RoverConfig("http://cam/stream", "ws://servo", "ws://motor")
    stream = VisionStream("http://cam/stream", cfg)
    buffer = bytearray()
    frame_one = b"\xff\xd8one\xff\xd9"
    frame_two = b"\xff\xd8two\xff\xd9"
    chunk = b"noise" + frame_one + b"junk" + frame_two + b"tail"

    stream._consume_mjpeg_chunk(chunk, buffer)

    assert stream.get_latest_frame() == frame_two


def test_detects_jpeg_snapshot_transport():
    stream = VisionStream("http://camera/cam-hi.jpg", RoverConfig("http://camera/cam-hi.jpg", "ws://servo", "ws://motor"))

    assert stream._transport == "jpeg_snapshot"


def test_builds_snapshot_fallback_candidates_for_http_stream():
    stream = VisionStream("http://camera:81/stream", RoverConfig("http://camera:81/stream", "ws://servo", "ws://motor"))

    assert "http://camera:81/cam-hi.jpg" in stream._snapshot_candidates
    assert "http://camera:81/cam-lo.jpg" in stream._snapshot_candidates


def test_consume_mjpeg_chunk_extracts_complete_frame():
    stream = VisionStream("http://localhost:81/stream", RoverConfig("http://localhost:81/stream", "ws://servo", "ws://motor"))

    payload = b"\xff\xd8" + b"abc123" + b"\xff\xd9"
    chunk = b"noise" + payload + b"tail"
    stream._consume_mjpeg_chunk(chunk, bytearray())

    latest = stream.get_latest_frame()
    assert latest == payload


def test_mjpeg_transport_prefers_raw_http_before_ffmpeg(monkeypatch):
    config = RoverConfig("http://localhost:81/stream", "ws://servo", "ws://motor")
    stream = VisionStream(config.vision_stream_url, config)
    calls: list[str] = []

    def fake_ffmpeg(_cv2):
        calls.append("ffmpeg")
        return False

    def fake_raw_http():
        calls.append("raw-http")
        stream._running = False
        return True

    monkeypatch.setattr(stream, "_run_mjpeg_ffmpeg_once", fake_ffmpeg)
    monkeypatch.setattr(stream, "_run_mjpeg_raw_http_once", fake_raw_http)
    monkeypatch.setattr(stream, "_run_snapshot_fallbacks_once", lambda: False)

    stream._running = True
    stream._run_mjpeg()

    assert calls == ["raw-http"]


def test_http_chunk_reader_prefers_read1_when_available():
    class DummyStream:
        def __init__(self):
            self.calls = []

        def read1(self, size):
            self.calls.append(("read1", size))
            return b"chunk"

        def read(self, size):
            self.calls.append(("read", size))
            return b"fallback"

    stream = DummyStream()

    chunk = VisionStream._read_http_chunk(stream, 8192)

    assert chunk == b"chunk"
    assert stream.calls == [("read1", 8192)]


def test_mjpeg_parser_discards_old_complete_frames_from_same_chunk(monkeypatch):
    cfg = RoverConfig("http://cam/stream", "ws://servo", "ws://motor")
    stream = VisionStream("http://cam/stream", cfg)
    ingested = []
    monkeypatch.setattr(stream, "_ingest_frame", lambda payload: ingested.append(payload))
    buffer = bytearray()
    frame_one = b"\xff\xd8old\xff\xd9"
    frame_two = b"\xff\xd8new\xff\xd9"

    stream._consume_mjpeg_chunk(frame_one + frame_two, buffer)

    assert ingested == [frame_two]
