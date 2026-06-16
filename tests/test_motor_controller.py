import json
from types import SimpleNamespace

from config import RoverConfig
from modules.motor_controller import MotorController, PendingMotorCommand


def decode_packet(payload: str) -> dict:
    return json.loads(payload)


class DummyUdpSocket:
    def __init__(self):
        self.sent = []

    def sendto(self, payload: bytes, target):
        self.sent.append((payload.decode("utf-8"), target))


def test_motor_controller_disables_on_404_error():
    controller = MotorController("ws://motor", RoverConfig("ws://cam", "ws://servo", "ws://motor"))

    controller._on_error(None, "Handshake status 404 Not Found")

    assert controller.is_connected() is False
    assert controller._disabled is True


def test_motor_controller_starts_disabled_when_url_missing():
    controller = MotorController("", RoverConfig("ws://cam", "ws://servo", ""))

    controller.start()

    assert controller.is_connected() is False
    assert controller._disabled is True


def test_motor_controller_translates_drive_command_to_legacy_csv_by_default():
    controller = MotorController(
        "ws://motor",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", motor_drive_speed=170),
    )
    recorded = []
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda command: recorded.append(command),
    )

    ok = controller._send_payloads(controller._payloads_for_command("F"), "F", 0.0)

    assert ok is True
    assert recorded == ["L,170", "R,170"]


def test_motor_controller_translates_turn_and_stop_commands_to_legacy_csv():
    controller = MotorController(
        "ws://motor",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", motor_turn_speed=150),
    )
    recorded = []
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda command: recorded.append(command),
    )

    assert controller._send_payloads(controller._payloads_for_command("L"), "L", 0.0) is True
    assert controller._send_payloads(controller._payloads_for_command("S"), "S", 0.0) is True

    assert recorded == ["L,-150", "R,150", "L,0", "R,0"]


def test_motor_controller_sends_udp_datagrams_for_legacy_packets():
    controller = MotorController(
        "udp://192.168.0.101:4210",
        RoverConfig("ws://cam", "udp://192.168.0.101:4210", "udp://192.168.0.101:4210"),
    )
    udp_socket = DummyUdpSocket()
    controller._udp_socket = udp_socket
    controller._connected = True

    ok = controller._send_payloads(controller._payloads_for_command("F"), "F", 0.0)

    assert ok is True
    assert udp_socket.sent == [
        ("L,170", ("192.168.0.101", 4210)),
        ("R,170", ("192.168.0.101", 4210)),
    ]


def test_motor_controller_send_queues_latest_command_without_direct_io():
    controller = MotorController("ws://motor", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    ok = controller.send("F")

    assert ok is True
    assert controller._pending_command is not None
    assert controller._pending_command.command == "F"
    assert controller._pending_command.payloads == ("L,170", "R,170")


def test_motor_controller_preserves_quick_tap_move_then_stop_order():
    controller = MotorController("ws://motor", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    controller._started = True

    assert controller.send("F") is True
    assert controller.send("S") is True

    first = controller._take_pending_command()
    second = controller._take_pending_command()

    assert first is not None
    assert second is not None
    assert first.command == "F"
    assert first.payloads == ("L,170", "R,170")
    assert second.command == "S"
    assert second.payloads == ("L,0", "R,0")


def test_motor_controller_force_stop_clears_pending_motion_and_sends_stop():
    controller = MotorController("ws://motor", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    recorded = []
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda command: recorded.append(command),
    )
    controller._connected = True
    controller._started = True
    controller.send("F")

    assert controller.force_stop() is True

    assert controller._pending_command is None
    assert controller._pending_followup_command is None
    assert recorded == ["L,0", "R,0"]


def test_motor_controller_drops_stale_motion_but_never_stop(monkeypatch):
    controller = MotorController(
        "ws://motor",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", motor_command_ttl_seconds=0.35),
    )
    monkeypatch.setattr("modules.motor_controller.time.monotonic", lambda: 10.0)

    stale_forward = PendingMotorCommand("F", controller._payloads_for_command("F"), 9.0)
    stale_stop = PendingMotorCommand("S", controller._payloads_for_command("S"), 9.0)

    assert controller._is_stale_motion_command(stale_forward) is True
    assert controller._is_stale_motion_command(stale_stop) is False


def test_motor_controller_sends_stop_when_socket_opens():
    controller = MotorController("ws://motor", RoverConfig("ws://cam", "ws://servo", "ws://motor"))
    recorded = []
    controller._app = SimpleNamespace(
        sock=SimpleNamespace(connected=True),
        send=lambda command: recorded.append(command),
    )

    controller._on_open(None)

    assert controller.is_connected() is True
    assert recorded == ["L,0", "R,0"]


def test_motor_controller_can_emit_legacy_csv_for_old_firmware():
    controller = MotorController(
        "ws://motor",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", transport_protocol="legacy_csv"),
    )

    assert controller._payloads_for_command("F") == ("L,170", "R,170")


def test_motor_controller_can_emit_hybrid_packets_for_json_firmware_transition():
    controller = MotorController(
        "ws://motor",
        RoverConfig("ws://cam", "ws://servo", "ws://motor", transport_protocol="hybrid"),
    )

    payloads = controller._payloads_for_command("F")

    assert decode_packet(payloads[0]) == {
        "cmd": "move",
        "dir": "F",
        "left": 170,
        "right": 170,
        "led": "blue",
    }
    assert payloads[1:] == ("L,170", "R,170")
