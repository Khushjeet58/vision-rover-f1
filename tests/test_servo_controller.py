import json
from types import SimpleNamespace

from config import RoverConfig
from modules.servo_controller import PendingServoCommand, ServoController


def decode_packet(payload: str) -> dict:
    return json.loads(payload)


class DummyUdpSocket:
    def __init__(self):
        self.sent = []

    def sendto(self, payload: bytes, target):
        self.sent.append((payload.decode("utf-8"), target))


def test_send_immediately_writes_to_connected_socket():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    recorded = []
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda command: recorded.append(command),
    )

    ok = controller._send_immediately(PendingServoCommand("Pan,120\nTilt,75", 0.0))

    assert ok is True
    assert recorded == ["Pan,120", "Tilt,75"]


def test_send_immediately_suppresses_errors_and_returns_false():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda _command: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    ok = controller._send_immediately(PendingServoCommand("Pan,90\nTilt,90", 0.0))

    assert ok is False


def test_send_immediately_writes_udp_datagrams_for_legacy_packets():
    controller = ServoController(
        "udp://192.168.0.101:4210",
        RoverConfig("ws://cam", "udp://192.168.0.101:4210", "udp://192.168.0.101:4210"),
    )
    udp_socket = DummyUdpSocket()
    controller._udp_socket = udp_socket
    controller._connected = True

    ok = controller._send_immediately(PendingServoCommand("Pan,120\nTilt,75", 0.0))

    assert ok is True
    assert udp_socket.sent == [
        ("Pan,120", ("192.168.0.101", 4210)),
        ("Tilt,75", ("192.168.0.101", 4210)),
    ]


def test_send_pan_tilt_updates_cached_angles_and_pending_payload():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    ok = controller.send_pan_tilt(120, 45)

    assert ok is True
    assert controller.current_angles() == (120, 45)
    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Pan,120\nTilt,45"


def test_send_pan_tilt_clamps_tilt_to_axis_specific_limit():
    controller = ServoController(
        "ws://servo",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", servo_tilt_max_angle=155),
    )
    controller._started = True

    ok = controller.send_pan_tilt(170, 180)

    assert ok is True
    assert controller.current_angles() == (170, 155)
    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Pan,170\nTilt,155"


def test_servo_controller_starts_at_configured_center_angle():
    controller = ServoController(
        "ws://servo",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", servo_center_angle=90),
    )

    assert controller.current_angles() == (90, 90)


def test_send_pan_tilt_deduplicates_same_pending_payload():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    first = controller.send_pan_tilt(120, 45)
    second = controller.send_pan_tilt(120, 45)

    assert first is True
    assert second is True
    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Pan,120\nTilt,45"


def test_send_updates_cached_angles_for_single_axis_payloads():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    controller.send("Pan,135")
    controller.send("Tilt,88")

    assert controller.current_angles() == (135, 88)
    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Tilt,88"


def test_on_open_requeues_current_pose_for_resync():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True
    controller.send_pan_tilt(123, 91)

    controller._on_open(None)

    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Pan,123\nTilt,91"


def test_on_open_does_not_center_servos_before_first_pose_command():
    controller = ServoController("ws://servo", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    controller._on_open(None)

    assert controller._pending_command is None


def test_servo_controller_starts_disabled_when_url_missing():
    controller = ServoController("", RoverConfig("ws://cam", "", "ws://motor"))

    controller.start()

    assert controller.is_connected() is False
    assert controller._disabled is True


def test_servo_controller_can_emit_legacy_csv_for_old_firmware():
    controller = ServoController(
        "ws://servo",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", transport_protocol="legacy_csv"),
    )
    controller._started = True

    controller.send_pan_tilt(120, 45)

    assert controller._pending_command is not None
    assert controller._pending_command.payload == "Pan,120\nTilt,45"


def test_servo_controller_can_emit_hybrid_packets_for_json_firmware_transition():
    controller = ServoController(
        "ws://servo",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", transport_protocol="hybrid"),
    )
    controller._started = True

    controller.send_pan_tilt(120, 45)

    assert controller._pending_command is not None
    payloads = controller._pending_command.payload.splitlines()
    assert decode_packet(payloads[0]) == {
        "cmd": "move",
        "pan": 120,
        "tilt": 45,
        "led": "blue",
    }
    assert payloads[1:] == ["Pan,120", "Tilt,45"]
