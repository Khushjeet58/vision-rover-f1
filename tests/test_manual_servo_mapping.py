from main import manual_servo_delta


class DummyVisionApp:
    def __init__(self):
        self.calls = []

    def engage_autonomous_target_lock(self):
        self.calls.append("engage_autonomous_target_lock")
        return type("Mode", (), {"value": "AUTONOMOUS"})()

    def toggle_autonomous_mode(self):
        self.calls.append("toggle_autonomous_mode")
        return type("Mode", (), {"value": "AUTONOMOUS"})()


def test_autonomous_button_command_engages_target_lock(monkeypatch):
    from main import JarvisSystem, MainController

    assert MainController is JarvisSystem
    controller = JarvisSystem.__new__(JarvisSystem)
    controller.rover_vision_app = DummyVisionApp()

    controller._handle_raw_command("__ENGAGE_AUTONOMOUS__")

    assert controller.rover_vision_app.calls == ["engage_autonomous_target_lock"]


def test_manual_servo_arrows_support_normal_direction_deltas():
    assert manual_servo_delta("__PAN_LEFT__", 5) == (-5, 0)
    assert manual_servo_delta("__PAN_RIGHT__", 5) == (5, 0)
    assert manual_servo_delta("__TILT_UP__", 5) == (0, -5)
    assert manual_servo_delta("__TILT_DOWN__", 5) == (0, 5)


def test_manual_servo_arrows_support_inverted_physical_mount():
    kwargs = {"pan_direction": -1, "tilt_direction": -1}

    assert manual_servo_delta("__PAN_LEFT__", 5, **kwargs) == (5, 0)
    assert manual_servo_delta("__PAN_RIGHT__", 5, **kwargs) == (-5, 0)
    assert manual_servo_delta("__TILT_UP__", 5, **kwargs) == (0, 5)
    assert manual_servo_delta("__TILT_DOWN__", 5, **kwargs) == (0, -5)


def test_voice_capture_routes_text_into_request_handler():
    from main import JarvisSystem

    controller = JarvisSystem.__new__(JarvisSystem)
    seen = []
    controller.handle_request = lambda text, is_raw_command=False: seen.append((text, is_raw_command))

    controller._handle_voice_text_captured("follow me")

    assert seen == [("follow me", False)]
