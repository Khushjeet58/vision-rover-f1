import numpy as np

from config import RoverConfig
from modules.detection_engine import DetectionBackend, DetectionEngine
from modules.rover_types import BoundingBox, Detection


class FakeBackend(DetectionBackend):
    def __init__(self, detections):
        self.detections = detections
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        assert frame.shape == (8, 8, 3)
        return list(self.detections)


def make_detection(label: str, area_scale: int) -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        bbox=BoundingBox(x=0, y=0, w=area_scale, h=area_scale, confidence=0.9),
    )


def test_detection_engine_uses_backend():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", face_lock_enabled=False)
    backend = FakeBackend([make_detection("person", 12)])
    engine = DetectionEngine(cfg, backend=backend)
    engine.load()
    detections = engine.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert backend.loaded is True
    assert len(detections) == 1
    assert detections[0].label == "person"


def test_select_primary_prefers_largest_target_label():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", face_lock_enabled=False)
    detections = [
        make_detection("person", 8),
        make_detection("bottle", 20),
        make_detection("person", 14),
    ]
    engine = DetectionEngine(cfg, backend=FakeBackend(detections))
    primary = engine.select_primary(detections)
    assert primary is not None
    assert primary.label == "person"
    assert primary.area == 196


def test_detection_engine_suppresses_nested_duplicate_target_boxes():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", face_lock_enabled=False)
    detections = [
        Detection(label="person", confidence=0.82, bbox=BoundingBox(0, 0, 100, 180, confidence=0.82)),
        Detection(label="person", confidence=0.91, bbox=BoundingBox(28, 24, 34, 46, confidence=0.91)),
    ]
    engine = DetectionEngine(cfg, backend=FakeBackend(detections))

    result = engine.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].bbox.w == 100
    assert result[0].bbox.h == 180


def test_detection_engine_adds_opencv_face_detections():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", face_lock_enabled=True)
    engine = DetectionEngine(cfg, backend=FakeBackend([]))

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return [(3, 4, 12, 14)]

    engine._face_cascade = FakeCascade()
    engine._face_ready = True

    result = engine.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].label == "face"
    assert result[0].bbox.x == 3
    assert result[0].bbox.h == 14


def test_detection_engine_keeps_face_and_person_detections_together_when_available():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor", face_lock_enabled=True)
    backend = FakeBackend([make_detection("person", 12)])
    engine = DetectionEngine(cfg, backend=backend)

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return [(3, 4, 12, 14)]

    engine._face_cascade = FakeCascade()
    engine._face_cascades = [FakeCascade()]
    engine._face_ready = True

    result = engine.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    labels = sorted(item.label for item in result)
    assert labels == ["face", "person"]


def test_detection_engine_ignores_empty_frame_source():
    cfg = RoverConfig("ws://cam", "ws://servo", "ws://motor")
    backend = FakeBackend([make_detection("person", 12)])
    engine = DetectionEngine(cfg, backend=backend)

    result = engine.detect(np.array([], dtype=np.uint8))

    assert result == []
