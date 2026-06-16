from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from runtime_bootstrap import ensure_project_venv
from runtime_preload import preload_onnxruntime

ensure_project_venv(Path(__file__).resolve().parent)
preload_onnxruntime()

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtNetwork import QLocalServer, QLocalSocket
from PyQt5.QtWidgets import QApplication

from config import Config, rover_config
from core.event_bus import SystemEvents, bus
from core.intent_router import IntentRouter
from modules.ai_ollama import OllamaAIEngine
from modules.audio_service import AudioService
from modules.command_handler import CommandHandler
from modules.control_arbiter import ControlArbiter
from modules.knowledge_base import KnowledgeBase
from modules.memory import Memory
from modules.operator_assistant import OperatorAssistant
from modules.rover_vision_app import RoverVisionApp
from modules.scene_perception import ScenePerceptionService
from modules.system_control import SystemController
from modules.tts_engine import TTSEngine
from ui.jarvis_hud import JarvisHUD


class SingleInstanceGuard(QObject):
    activation_requested = pyqtSignal()

    def __init__(self, server_name: str):
        super().__init__()
        self._server_name = server_name
        self._server = QLocalServer(self)

    def acquire(self) -> bool:
        socket = QLocalSocket(self)
        socket.connectToServer(self._server_name)
        if socket.waitForConnected(150):
            socket.write(b"ACTIVATE")
            socket.flush()
            socket.waitForBytesWritten(150)
            socket.disconnectFromServer()
            return False

        QLocalServer.removeServer(self._server_name)
        if not self._server.listen(self._server_name):
            return False
        self._server.newConnection.connect(self._handle_connection)
        return True

    def _handle_connection(self) -> None:
        socket = self._server.nextPendingConnection()
        if socket is None:
            return
        socket.waitForReadyRead(150)
        _ = bytes(socket.readAll())
        socket.disconnectFromServer()
        self.activation_requested.emit()


class MainThreadBridge(QObject):
    activation_requested = pyqtSignal()

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.activation_requested.connect(self._handle_activation_requested)

    @pyqtSlot()
    def _handle_activation_requested(self) -> None:
        self._callback()


def _single_instance_enabled(argv: list[str]) -> bool:
    if "--allow-multi-instance" in argv:
        return False
    env_value = (os.getenv("VISION_ALLOW_MULTI_INSTANCE", "") or "").strip().lower()
    return env_value not in {"1", "true", "yes", "on"}


def manual_servo_delta(
    command: str,
    step: float,
    *,
    pan_direction: int = 1,
    tilt_direction: int = 1,
) -> tuple[float, float] | None:
    token = (command or "").strip().upper()
    pan_sign = 1 if pan_direction >= 0 else -1
    tilt_sign = 1 if tilt_direction >= 0 else -1
    if token == "__PAN_LEFT__":
        return -step * pan_sign, 0
    if token == "__PAN_RIGHT__":
        return step * pan_sign, 0
    if token == "__TILT_UP__":
        return 0, -step * tilt_sign
    if token == "__TILT_DOWN__":
        return 0, step * tilt_sign
    return None


class JarvisSystem:
    def __init__(self):
        self.memory = Memory()
        self.intent_router = IntentRouter()
        self.system_controller = SystemController()
        self.control_arbiter = ControlArbiter()
        self.knowledge_base = KnowledgeBase(rover_config)
        self.ai_engine = OllamaAIEngine(self.knowledge_base)
        self.scene_perception = ScenePerceptionService()
        self.operator_assistant = OperatorAssistant(self.knowledge_base)
        self.tts = TTSEngine()
        self.audio_service = AudioService(rover_config)
        self.rover_vision_app = RoverVisionApp(rover_config, self.control_arbiter)
        self._scene_voice_lock = threading.Lock()
        self._latest_target = None
        self._latest_observed_labels: tuple[str, ...] = ()
        self._last_detection_announcement_at = 0.0
        self._scene_person_visible = False
        self._scene_target_locked = False
        self._last_person_seen_at = 0.0
        self._scene_voice_inflight = False
        self._last_scene_signature = ""
        self._last_spoken_line = ""
        self._last_spoken_at = 0.0
        bus.subscribe(SystemEvents.DETECTIONS_UPDATED, self._handle_detections_updated)
        bus.subscribe(SystemEvents.TRACK_TARGET_CHANGED, self._handle_track_target_changed)
        bus.subscribe(SystemEvents.VOICE_TEXT_CAPTURED, self._handle_voice_text_captured)
        self.audio_service.toggle_listening(True)

        threading.Thread(
            target=self.rover_vision_app.run,
            daemon=True,
            name="RoverVisionRuntime_Thread",
        ).start()

    def stop(self):
        self.audio_service.stop()
        self.rover_vision_app.stop()

    def _handle_voice_text_captured(self, transcript: str) -> None:
        text = (transcript or "").strip()
        if not text:
            return
        self.handle_request(text)

    def handle_request(self, user_input: str, is_raw_command: bool = False):
        if not user_input:
            return

        if is_raw_command:
            self._handle_raw_command(user_input)
            return

        text = user_input.strip()
        if not text:
            return

        bus.emit(SystemEvents.LOG_MESSAGE, f"> Route analyzing: '{text}'")
        bus.emit(SystemEvents.STATE_CHANGE, "THINKING")

        local_cmd = CommandHandler.parse_local_command(text)
        if local_cmd:
            self._execute_control_command(local_cmd, source="VOICE")
            return

        intent = self.intent_router.detect_intent(text)
        if intent == IntentRouter.SYSTEM:
            self._handle_system(text)
            return

        self._handle_chat(text)

    def _handle_raw_command(self, command: str) -> None:
        token = (command or "").strip().upper()
        if token in {"F", "B", "L", "R", "S"}:
            self.rover_vision_app.send_drive_command(token, source="KEYBOARD")
            return

        servo_delta = manual_servo_delta(
            token,
            rover_config.servo_step,
            pan_direction=rover_config.servo_manual_pan_direction,
            tilt_direction=rover_config.servo_manual_tilt_direction,
        )
        if servo_delta is not None:
            pan_delta, tilt_delta = servo_delta
            self.rover_vision_app.adjust_servo(pan_delta=pan_delta, tilt_delta=tilt_delta, source="KEYBOARD")
        elif token == "__TOGGLE_FOLLOW__":
            mode = self.rover_vision_app.toggle_follow_mode()
            bus.emit(SystemEvents.LOG_MESSAGE, f"[Control] Mode set to {mode.value}")
        elif token == "__TOGGLE_AUTONOMOUS__":
            mode = self.rover_vision_app.toggle_autonomous_mode()
            bus.emit(SystemEvents.LOG_MESSAGE, f"[Control] Mode set to {mode.value}")
        elif token == "__ENGAGE_AUTONOMOUS__":
            mode = self.rover_vision_app.engage_autonomous_target_lock()
            bus.emit(SystemEvents.LOG_MESSAGE, f"[Control] Mode set to {mode.value}")
        elif token == "__INSPECT_SCENE__":
            self._execute_control_command("INSPECT", source="VOICE")
        elif token == "__E_STOP__":
            self._execute_control_command("E_STOP", source="E_STOP")

    def _execute_control_command(self, command: str, source: str) -> None:
        cmd = (command or "").upper()

        if cmd in {"F", "B", "L", "R", "S"}:
            self.memory.update_command(cmd)
            self.rover_vision_app.send_drive_command(cmd, source=source)
            self._speak(CommandHandler.speech_for(cmd), interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        if cmd == "FOLLOW":
            self.rover_vision_app.set_follow_mode()
            self._speak(CommandHandler.speech_for(cmd), interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        if cmd == "AUTO":
            self.rover_vision_app.set_autonomous_mode()
            self._speak(CommandHandler.speech_for(cmd), interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        if cmd == "MANUAL":
            self.rover_vision_app.set_manual_mode()
            self._speak(CommandHandler.speech_for(cmd), interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        if cmd == "INSPECT":
            self.control_arbiter.begin_scene_inspection()
            description = self.rover_vision_app.describe_scene()
            self._speak(description, interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        if cmd == "E_STOP":
            self.rover_vision_app.emergency_stop()
            self._speak(CommandHandler.speech_for(cmd), interrupt=True)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        bus.emit(SystemEvents.STATE_CHANGE, "IDLE")

    def _handle_system(self, text: str) -> None:
        result = self.system_controller.handle_text(text)
        if result.get("speech"):
            self._speak(result["speech"])
        bus.emit(SystemEvents.STATE_CHANGE, "IDLE")

    def _handle_chat(self, text: str) -> None:
        snapshot = self.rover_vision_app.latest_snapshot()
        local_answer = self.operator_assistant.try_answer(text, snapshot)
        if local_answer:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[V.I.S.I.O.N] {local_answer}")
            self._speak(local_answer)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            return

        def callback(response_text: str):
            bus.emit(SystemEvents.LOG_MESSAGE, f"[V.I.S.I.O.N] {response_text}")
            self._speak(response_text)
            bus.emit(SystemEvents.STATE_CHANGE, "IDLE")

        self.ai_engine.run_chat_query_async(
            text,
            callback,
            runtime_context=self.operator_assistant.build_runtime_context(snapshot),
        )

    def _handle_track_target_changed(self, target) -> None:
        with self._scene_voice_lock:
            self._latest_target = target
            labels = self._latest_observed_labels
        locked = target is not None and getattr(target, "stable_frames", 0) >= rover_config.target_lock_frames
        if not locked:
            with self._scene_voice_lock:
                self._scene_target_locked = False
            return
        if rover_config.target_label not in labels:
            return
        with self._scene_voice_lock:
            if self._scene_target_locked:
                return
            self._scene_target_locked = True
        self._queue_scene_voice(labels, locked=True, event="lock", target=target)

    def _handle_detections_updated(self, detections) -> None:
        detections = list(detections or [])
        now = time.monotonic()
        observed_labels = tuple(
            str(getattr(item, "label", "")).strip().lower()
            for item in detections
            if str(getattr(item, "label", "")).strip()
        )
        person_visible = rover_config.target_label in observed_labels
        if person_visible:
            self._last_person_seen_at = now

        locked = self._target_is_locked()
        with self._scene_voice_lock:
            self._latest_observed_labels = observed_labels
            if self._scene_person_visible and (now - self._last_person_seen_at) >= 1.2 and not person_visible:
                self._scene_person_visible = False
                self._scene_target_locked = False
                self._last_scene_signature = ""
                return

            if not self._scene_person_visible and person_visible:
                self._scene_person_visible = True
                self._scene_target_locked = locked
            elif not person_visible:
                self._scene_target_locked = False

        if person_visible:
            self._queue_scene_voice(observed_labels, locked=locked, event="enter")

    def _queue_scene_voice(
        self,
        labels: tuple[str, ...],
        *,
        locked: bool,
        event: str,
        target=None,
    ) -> None:
        now = time.monotonic()
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        target_key = getattr(target, "target_id", 0) or 0
        signature = "|".join(f"{label}:{counts[label]}" for label in sorted(counts))
        signature = f"{event}|{signature}|locked:{int(locked)}|target:{target_key}"
        min_gap = 2.5 if event == "lock" else max(6.0, rover_config.scene_announce_cooldown_seconds)

        with self._scene_voice_lock:
            if not labels:
                return
            if self._scene_voice_inflight:
                return
            if signature == self._last_scene_signature:
                return
            if (now - self._last_detection_announcement_at) < min_gap:
                return
            self._scene_voice_inflight = True
            self._last_scene_signature = signature
            self._last_detection_announcement_at = now

        self.ai_engine.run_scene_update_async(
            list(labels),
            locked=locked,
            callback=lambda response, sig=signature, observed=labels, is_locked=locked: self._handle_scene_voice_response(
                sig,
                observed,
                is_locked,
                response,
            ),
        )

    def _handle_scene_voice_response(
        self,
        signature: str,
        observed_labels: tuple[str, ...],
        locked: bool,
        response: str,
    ) -> None:
        with self._scene_voice_lock:
            self._scene_voice_inflight = False
            current_visible = self._scene_person_visible
            current_signature = self._last_scene_signature
        if not current_visible or signature != current_signature:
            return
        line = response or self.scene_perception.live_scene_line(list(observed_labels), locked=locked)
        if not line:
            return
        self._speak(line, allow_when_busy=False)

    def _target_is_locked(self) -> bool:
        with self._scene_voice_lock:
            target = self._latest_target
        if target is None:
            return False
        return getattr(target, "stable_frames", 0) >= rover_config.target_lock_frames

    def _speak(self, text: str, interrupt: bool = False, allow_when_busy: bool = True):
        line = text.strip() if text and text.strip() else "Sorry, I did not catch that properly."
        now = time.monotonic()
        if not interrupt and line == self._last_spoken_line and (now - self._last_spoken_at) < 2.0:
            return
        if not interrupt and not allow_when_busy and self.tts.has_pending():
            return
        try:
            self.tts.speak(
                line,
                on_start=lambda: (
                    bus.emit(SystemEvents.TTS_STARTED, line),
                    bus.emit(SystemEvents.STATE_CHANGE, "SPEAKING"),
                ),
                on_done=lambda: (
                    bus.emit(SystemEvents.TTS_FINISHED, line),
                    bus.emit(SystemEvents.STATE_CHANGE, "IDLE"),
                ),
                interrupt=interrupt,
            )
            self._last_spoken_line = line
            self._last_spoken_at = now
        except Exception as exc:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[TTS] Speech failed: {exc}")


def main():
    app = QApplication(sys.argv)

    use_single_instance = _single_instance_enabled(sys.argv[1:])
    guard = SingleInstanceGuard(Config.SINGLE_INSTANCE_SERVER) if use_single_instance else None
    if guard is not None and not guard.acquire():
        print(
            "[V.I.S.I.O.N] Existing instance is already running. "
            "Close it first to load new code changes, or relaunch with "
            "--allow-multi-instance (or VISION_ALLOW_MULTI_INSTANCE=1)."
        )
        return 0

    controller = JarvisSystem()
    window = JarvisHUD(request_handler_callback=controller.handle_request, config=rover_config)
    window.show()

    def activate_window():
        window.showNormal()
        window.raise_()
        window.activateWindow()

    bridge = MainThreadBridge(activate_window)
    controller.audio_service.set_launch_callback(bridge.activation_requested.emit)

    if guard is not None:
        guard.activation_requested.connect(activate_window)

    ret = app.exec_()
    controller.stop()
    return ret


if __name__ == "__main__":
    sys.exit(main())


MainController = JarvisSystem
