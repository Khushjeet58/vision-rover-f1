import numpy as np

from modules.operator_assistant import OperatorAssistant
from modules.rover_types import BoundingBox, ConnectionState, ControlMode, Detection, VisionSnapshot


def make_snapshot(**overrides) -> VisionSnapshot:
    snapshot = VisionSnapshot(
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
        detections=[],
        mode=ControlMode.MANUAL,
        links={"camera": ConnectionState.CONNECTED, "servo": ConnectionState.CONNECTED},
    )
    for key, value in overrides.items():
        setattr(snapshot, key, value)
    return snapshot


def test_operator_assistant_answers_scene_question_locally():
    assistant = OperatorAssistant()
    snapshot = make_snapshot(
        detections=[
            Detection(label="person", confidence=0.9, bbox=BoundingBox(0, 0, 20, 20)),
            Detection(label="bottle", confidence=0.8, bbox=BoundingBox(24, 0, 8, 16)),
        ]
    )

    response = assistant.try_answer("what do you see right now", snapshot)

    assert response is not None
    assert "person" in response.lower()


def test_operator_assistant_answers_autonomy_status():
    assistant = OperatorAssistant()
    snapshot = make_snapshot(mode=ControlMode.AUTONOMOUS)

    response = assistant.try_answer("what mode are you in", snapshot)

    assert response == "I am in autonomous mode and driving from the live scene feed."


def test_operator_assistant_builds_runtime_context():
    assistant = OperatorAssistant()
    snapshot = make_snapshot(
        detections=[Detection(label="person", confidence=0.9, bbox=BoundingBox(10, 10, 20, 20))],
        last_command="F",
        servo_pan=105,
        servo_tilt=82,
    )

    context = assistant.build_runtime_context(snapshot)

    assert "mode=MANUAL" in context
    assert "detections=person:1" in context
    assert "last_command=F" in context


def test_operator_assistant_answers_command_help():
    assistant = OperatorAssistant()

    response = assistant.try_answer("what commands can I say", make_snapshot())

    assert response is not None
    assert "move forward" in response.lower()
    assert "follow me" in response.lower()


def test_operator_assistant_answers_project_question():
    assistant = OperatorAssistant()

    response = assistant.try_answer("what is this project and how does it work", make_snapshot())

    assert response is not None
    assert "esp32-cam rover" in response.lower()
    assert "vision" in response.lower() or "v.i.s.i.o.n." in response.lower()
