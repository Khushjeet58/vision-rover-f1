from __future__ import annotations

import threading
import time

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QApplication,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import RoverConfig
from core.event_bus import SystemEvents, bus
from modules.rover_types import ConnectionState
from ui.arc_reactor_widget import ArcReactorWidget
from ui.theme import JARVIS_THEME


class UIEventBridge(QObject):
    log_signal = pyqtSignal(str)
    state_signal = pyqtSignal(str)
    voice_captured_signal = pyqtSignal(str)
    cmd_executed_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(object)
    mode_signal = pyqtSignal(str)
    connection_signal = pyqtSignal(object)

    def __init__(self, frame_hz: int = 30):
        super().__init__()
        self._frame_lock = threading.Lock()
        self._pending_frame = None
        bus.subscribe(SystemEvents.LOG_MESSAGE, self.log_signal.emit)
        bus.subscribe(SystemEvents.STATE_CHANGE, self.state_signal.emit)
        bus.subscribe(SystemEvents.VOICE_TEXT_CAPTURED, self.voice_captured_signal.emit)
        bus.subscribe(SystemEvents.COMMAND_EXECUTED, self.cmd_executed_signal.emit)
        bus.subscribe(SystemEvents.FRAME_READY, self._queue_frame)
        bus.subscribe(SystemEvents.CONTROL_MODE_CHANGED, self.mode_signal.emit)
        bus.subscribe(SystemEvents.CONNECTION_STATUS_CHANGED, self.connection_signal.emit)

        self._frame_timer = QTimer(self)
        interval_ms = max(10, int(round(1000 / max(1, frame_hz))))
        self._frame_timer.setInterval(interval_ms)
        self._frame_timer.timeout.connect(self._flush_frame)
        self._frame_timer.start()

    def _queue_frame(self, snapshot) -> None:
        with self._frame_lock:
            self._pending_frame = snapshot

    def _flush_frame(self) -> None:
        with self._frame_lock:
            snapshot = self._pending_frame
            self._pending_frame = None
        if snapshot is not None:
            self.frame_signal.emit(snapshot)


class CameraFeedWidget(QLabel):
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self._placeholder = placeholder
        self._base_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(self.FRAME_WIDTH, self.FRAME_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if width <= 0:
            return self.FRAME_HEIGHT
        return int(width * self.FRAME_HEIGHT / self.FRAME_WIDTH)

    def show_placeholder(self, text: str | None = None) -> None:
        self._base_pixmap = None
        self.clear()
        self.setText(text or self._placeholder)

    def set_frame_image(self, image: QImage) -> None:
        self._base_pixmap = QPixmap.fromImage(image.copy())
        self._apply_viewport_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_viewport_pixmap()

    def _apply_viewport_pixmap(self) -> None:
        if self._base_pixmap is None:
            return
        viewport = self.contentsRect().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        transform_mode = (
            Qt.SmoothTransformation
            if self._base_pixmap.width() < viewport.width() or self._base_pixmap.height() < viewport.height()
            else Qt.FastTransformation
        )
        scaled = self._base_pixmap.scaled(
            viewport,
            Qt.KeepAspectRatio,
            transform_mode,
        )
        self.clear()
        self.setPixmap(scaled)


class JarvisHUD(QMainWindow):
    DRIVE_KEY_MAP = {
        Qt.Key_W: "F",
        Qt.Key_S: "B",
        Qt.Key_A: "L",
        Qt.Key_D: "R",
    }

    SERVO_KEY_MAP = {
        Qt.Key_Left: "__PAN_LEFT__",
        Qt.Key_Right: "__PAN_RIGHT__",
        Qt.Key_Up: "__TILT_UP__",
        Qt.Key_Down: "__TILT_DOWN__",
    }

    NATIVE_SERVO_KEY_MAP = {
        37: "__PAN_LEFT__",
        39: "__PAN_RIGHT__",
        38: "__TILT_UP__",
        40: "__TILT_DOWN__",
    }

    NATIVE_SCAN_SERVO_KEY_MAP = {
        72: "__TILT_UP__",
        75: "__PAN_LEFT__",
        77: "__PAN_RIGHT__",
        80: "__TILT_DOWN__",
        328: "__TILT_UP__",
        331: "__PAN_LEFT__",
        333: "__PAN_RIGHT__",
        336: "__TILT_DOWN__",
    }

    def __init__(self, request_handler_callback, config: RoverConfig | None = None):
        super().__init__()
        self.request_handler = request_handler_callback
        self._config = config
        frame_hz = config.ui_frame_hz if config is not None else 30
        self.bridge = UIEventBridge(frame_hz=frame_hz)

        self._pressed_drive_keys: set[int] = set()
        self._pressed_servo_keys: set[str] = set()
        self._key_timestamps: dict[object, int] = {}
        self._key_counter = 0
        self._last_drive_command = "S"
        self._last_key_debug_at = 0.0

        self._build_ui()
        self.setStyleSheet(JARVIS_THEME)

        self.bridge.log_signal.connect(self._append_log)
        self.bridge.state_signal.connect(self._update_core_state)
        self.bridge.voice_captured_signal.connect(self._handle_voice_input)
        self.bridge.cmd_executed_signal.connect(self._update_telemetry)
        self.bridge.frame_signal.connect(self._update_frame)
        self.bridge.mode_signal.connect(self._update_mode)
        self.bridge.connection_signal.connect(self._update_connections)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        QTimer.singleShot(0, lambda: self.setFocus(Qt.OtherFocusReason))

        self._input_timer = QTimer(self)
        key_repeat_hz = self._config.key_repeat_hz if self._config is not None else 15
        self._input_timer.setInterval(max(16, int(round(1000 / max(1, key_repeat_hz)))))
        self._input_timer.timeout.connect(self._dispatch_held_keys)
        self._input_timer.start()

    def _build_ui(self) -> None:
        self.setWindowTitle("V.I.S.I.O.N Control Panel")
        self.resize(1560, 960)
        self.setFocusPolicy(Qt.StrongFocus)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title = QLabel("V.I.S.I.O.N CONTROL PANEL")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Local AI rover operator for manual drive, autonomous mode, follow mode, and scene awareness")
        subtitle.setObjectName("subtitleLabel")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch(1)

        self.mode_label = QLabel("IDLE")
        self.mode_label.setObjectName("modeBadge")
        header.addWidget(self.mode_label)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(14)
        content.addWidget(self._build_left_column(), 1)
        content.addWidget(self._build_right_column(), 2)
        root.addLayout(content, 1)

        root.addWidget(self._build_input_panel())

    def _build_left_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(14)
        layout.addWidget(self._build_core_panel())
        layout.addWidget(self._build_status_panel())
        layout.addStretch(1)
        return column

    def _build_right_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setSpacing(14)
        layout.addWidget(self._build_camera_panel(), 2)
        layout.addWidget(self._build_console_panel(), 1)
        return column

    def _build_core_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("heroPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("COMMAND CORE")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.reactor = ArcReactorWidget()
        layout.addWidget(self.reactor, 1, alignment=Qt.AlignCenter)

        chips = QHBoxLayout()
        self.camera_chip = QLabel("CAM OFFLINE")
        self.motor_chip = QLabel("MOTOR OFFLINE")
        self.servo_chip = QLabel("SERVO OFFLINE")
        self.detector_chip = QLabel("YOLO OFFLINE")
        self.ollama_chip = QLabel("OLLAMA OFFLINE")
        for widget in (self.camera_chip, self.motor_chip, self.servo_chip, self.detector_chip, self.ollama_chip):
            widget.setObjectName("chip")
            chips.addWidget(widget)
        layout.addLayout(chips)

        actions = QHBoxLayout()
        self.auto_btn = QPushButton("AUTO LOCK")
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(lambda: self.request_handler("__ENGAGE_AUTONOMOUS__", is_raw_command=True))
        self.follow_btn = QPushButton("FOLLOW")
        self.follow_btn.setCheckable(True)
        self.follow_btn.clicked.connect(lambda: self.request_handler("__TOGGLE_FOLLOW__", is_raw_command=True))
        self.estop_btn = QPushButton("E-STOP")
        self.estop_btn.setObjectName("dangerButton")
        self.estop_btn.clicked.connect(lambda: self.request_handler("__E_STOP__", is_raw_command=True))
        actions.addWidget(self.auto_btn)
        actions.addWidget(self.follow_btn)
        actions.addWidget(self.estop_btn)
        layout.addLayout(actions)
        return panel

    def _build_status_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QGridLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)

        title = QLabel("ROVER STATUS")
        title.setObjectName("panelTitle")
        layout.addWidget(title, 0, 0, 1, 2)

        self.motion_value = QLabel("STOPPED")
        self.last_cmd_value = QLabel("S")
        self.fps_value = QLabel("0.0")
        self.target_value = QLabel("NONE")
        self.servo_value = QLabel("090 / 090")
        self.latency_value = QLabel("0.0 ms")

        metrics = [
            ("MOTION", self.motion_value),
            ("LAST CMD", self.last_cmd_value),
            ("FPS R/C", self.fps_value),
            ("TARGET", self.target_value),
            ("SERVO", self.servo_value),
            ("NET LAT", self.latency_value),
        ]
        for row, (label_text, value_label) in enumerate(metrics, start=1):
            key = QLabel(label_text)
            key.setObjectName("metricKey")
            value_label.setObjectName("metricValue")
            layout.addWidget(key, row, 0)
            layout.addWidget(value_label, row, 1)
        return panel

    def _build_camera_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("heroPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("LIVE VISION")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        hud_strip = QHBoxLayout()
        hud_strip.setSpacing(8)
        self.camera_mode_badge = QLabel("IDLE")
        self.camera_mode_badge.setObjectName("cameraHudBadgeStrong")
        self.camera_cmd_badge = QLabel("CMD S")
        self.camera_cmd_badge.setObjectName("cameraHudBadge")
        self.camera_fps_badge = QLabel("FPS 0.0 / 0.0")
        self.camera_fps_badge.setObjectName("cameraHudBadge")
        self.camera_ai_badge = QLabel("AI 0.0 ms")
        self.camera_ai_badge.setObjectName("cameraHudBadge")
        self.camera_link_badge = QLabel("CAM OFFLINE")
        self.camera_link_badge.setObjectName("cameraHudBadge")
        self.camera_target_badge = QLabel("TARGET NONE")
        self.camera_target_badge.setObjectName("cameraHudBadge")
        self.camera_servo_badge = QLabel("SERVO 090/090")
        self.camera_servo_badge.setObjectName("cameraHudBadge")
        self.camera_latency_badge = QLabel("NET 0.0 ms")
        self.camera_latency_badge.setObjectName("cameraHudBadge")
        for widget in (
            self.camera_mode_badge,
            self.camera_cmd_badge,
            self.camera_fps_badge,
            self.camera_ai_badge,
            self.camera_link_badge,
            self.camera_target_badge,
            self.camera_servo_badge,
            self.camera_latency_badge,
        ):
            hud_strip.addWidget(widget)
        hud_strip.addStretch(1)
        layout.addLayout(hud_strip)

        self.camera_feed = CameraFeedWidget("Waiting for ESP32-CAM stream...")
        self.camera_feed.setObjectName("cameraFeed")
        self.camera_feed.setFixedSize(760, 570)
        feed_row = QHBoxLayout()
        feed_row.addStretch(1)
        feed_row.addWidget(self.camera_feed, 1)
        feed_row.addStretch(1)
        layout.addLayout(feed_row, 1)

        self.camera_caption = QLabel(
            "640x480 optimized live view (4:3). W/A/S/D drive, arrows pan/tilt, P auto, T follow, Space E-stop"
        )
        self.camera_caption.setObjectName("cameraCaption")
        layout.addWidget(self.camera_caption)
        return panel

    def _build_console_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("MISSION LOG")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        layout.addWidget(self.console, 1)
        return panel

    def _build_input_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.mic_btn = QPushButton("MIC OFF")
        self.mic_btn.setCheckable(True)
        self.mic_btn.setChecked(True)
        self.mic_btn.setText("MIC LIVE")
        self.mic_btn.clicked.connect(self._toggle_mic)
        layout.addWidget(self.mic_btn)

        inspect_btn = QPushButton("INSPECT")
        inspect_btn.clicked.connect(lambda: self.request_handler("__INSPECT_SCENE__", is_raw_command=True))
        layout.addWidget(inspect_btn)

        self.input_box = QLineEdit()
        self.input_box.setFocusPolicy(Qt.ClickFocus)
        self.input_box.setPlaceholderText("Ask V.I.S.I.O.N to explain the project, move the rover, or inspect the scene...")
        self.input_box.returnPressed.connect(self._submit_text)
        layout.addWidget(self.input_box, 1)

        send_btn = QPushButton("EXECUTE")
        send_btn.clicked.connect(self._submit_text)
        layout.addWidget(send_btn)
        return panel

    def _toggle_mic(self) -> None:
        is_on = self.mic_btn.isChecked()
        self.mic_btn.setText("MIC LIVE" if is_on else "MIC OFF")
        bus.emit(SystemEvents.MIC_TOGGLE, is_on)

    def _append_log(self, text: str) -> None:
        self.console.append(text)

    def _update_core_state(self, state: str) -> None:
        state_upper = (state or "IDLE").upper()
        if state_upper == "THINKING":
            self.reactor.set_thinking()
        elif state_upper == "SPEAKING":
            self.reactor.set_speaking()
        else:
            self.reactor.set_idle()

    def _update_mode(self, mode: str) -> None:
        mode = (mode or "IDLE").upper()
        self.mode_label.setText(mode)
        self.camera_mode_badge.setText(mode)
        self.auto_btn.setChecked(mode == "AUTONOMOUS")
        self.follow_btn.setChecked(mode == "FOLLOW_PERSON")

    def _update_connections(self, status) -> None:
        if status is None:
            return
        channel = getattr(status, "channel", "")
        state = getattr(status, "state", None)
        state_text = getattr(state, "value", "UNKNOWN")
        detail = (getattr(status, "detail", "") or "").lower()
        self._apply_connection_state(channel, state_text, detail)

    def _update_telemetry(self, cmd: str) -> None:
        cmd = (cmd or "").upper()
        self.last_cmd_value.setText(cmd or "S")
        mapping = {
            "F": "FORWARD",
            "B": "REVERSE",
            "L": "TURN LEFT",
            "R": "TURN RIGHT",
            "S": "STOPPED",
        }
        self.motion_value.setText(mapping.get(cmd, cmd or "STOPPED"))
        self.camera_cmd_badge.setText(f"CMD {cmd or 'S'}")

    def _update_frame(self, snapshot) -> None:
        if snapshot is None:
            return
        links = getattr(snapshot, "links", None) or {}
        for channel, state in links.items():
            state_text = getattr(state, "value", "UNKNOWN")
            self._apply_connection_state(str(channel), state_text)
        frame = getattr(snapshot, "frame", None)
        render_fps = getattr(snapshot, "fps", 0.0)
        source_fps = getattr(snapshot, "source_fps", 0.0)
        inference_ms = getattr(snapshot, "inference_ms", 0.0)
        servo_pan = getattr(snapshot, "servo_pan", 90)
        servo_tilt = getattr(snapshot, "servo_tilt", 90)
        latency_ms = getattr(snapshot, "network_latency_ms", 0.0)
        self.fps_value.setText(f"{render_fps:.1f}/{source_fps:.1f}")
        self.camera_fps_badge.setText(f"FPS {render_fps:.1f} / {source_fps:.1f}")
        self.camera_ai_badge.setText(f"AI {inference_ms:.1f} ms")
        self.servo_value.setText(f"{servo_pan:03d} / {servo_tilt:03d}")
        self.latency_value.setText(f"{latency_ms:.1f} ms")
        self.camera_servo_badge.setText(f"SERVO {servo_pan:03d}/{servo_tilt:03d}")
        self.camera_latency_badge.setText(f"NET {latency_ms:.1f} ms")
        target = getattr(snapshot, "target", None)
        target_coords = getattr(snapshot, "target_coords", None)
        if target_coords:
            self.target_value.setText(f"{target_coords[0]}, {target_coords[1]}")
        else:
            self.target_value.setText("NONE")
        if target:
            self.camera_target_badge.setText("TARGET LOCKED")
        else:
            self.camera_target_badge.setText("TARGET NONE")

        if frame is None:
            self.camera_feed.show_placeholder("Waiting for ESP32-CAM stream...")
            return

        height, width, _ = frame.shape
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_RGB888)
        self.camera_feed.set_frame_image(image)

    def _apply_connection_state(self, channel: str, state_text: str, detail: str = "") -> None:
        chip_text, badge_text = self.connection_labels(channel, state_text, detail)
        if channel == "camera":
            self.camera_chip.setText(chip_text)
            if badge_text is not None:
                self.camera_link_badge.setText(badge_text)
        elif channel == "motor":
            self.motor_chip.setText(chip_text)
        elif channel == "servo":
            self.servo_chip.setText(chip_text)
        elif channel == "detector":
            self.detector_chip.setText(chip_text)
        elif channel == "ollama":
            self.ollama_chip.setText(chip_text)

    @staticmethod
    def connection_labels(channel: str, state_text: str, detail: str = "") -> tuple[str, str | None]:
        state_text = (state_text or "UNKNOWN").upper()
        detail = (detail or "").lower()
        if channel == "camera":
            return f"CAM {state_text}", f"CAM {state_text}"
        if channel == "motor":
            return ("MOTOR STANDBY" if "disabled" in detail else f"MOTOR {state_text}"), None
        if channel == "servo":
            return ("SERVO STANDBY" if "disabled" in detail else f"SERVO {state_text}"), None
        if channel == "detector":
            return f"YOLO {state_text}", None
        if channel == "ollama":
            return f"OLLAMA {state_text}", None
        return f"{channel.upper()} {state_text}", None

    def _handle_voice_input(self, text: str) -> None:
        self.request_handler(text)

    def _submit_text(self) -> None:
        text = self.input_box.text().strip()
        if text:
            self.input_box.clear()
            self._append_log(f"[USER] {text}")
            self.request_handler(text)

    def _dispatch_held_keys(self) -> None:
        if not self.input_box.hasFocus():
            drive_key = self._most_recent_key(self._pressed_drive_keys)
            if drive_key is not None:
                command = self.DRIVE_KEY_MAP[drive_key]
                self.request_handler(command, is_raw_command=True)
                self._last_drive_command = command
            elif self._last_drive_command != "S":
                self.request_handler("S", is_raw_command=True)
                self._last_drive_command = "S"
            servo_command = self._most_recent_servo_command()
            if servo_command is not None:
                self.request_handler(servo_command, is_raw_command=True)

    def keyPressEvent(self, event) -> None:
        if self._handle_key_press(event):
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self._handle_key_release(event):
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if not self.isVisible():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.KeyPress:
            if self._handle_key_press(event):
                return True
        elif event.type() == QEvent.KeyRelease:
            if self._handle_key_release(event):
                return True
        return super().eventFilter(watched, event)

    def _most_recent_key(self, keys: set[int]) -> int | None:
        if not keys:
            return None
        return max(keys, key=lambda item: self._key_timestamps.get(item, 0))

    def _is_global_control_key(self, key: int) -> bool:
        return (
            key in self.SERVO_KEY_MAP
            or key in {Qt.Key_Space, Qt.Key_T, Qt.Key_P, Qt.Key_I, Qt.Key_M, Qt.Key_Escape}
        )

    def _handle_key_press(self, event) -> bool:
        key = event.key()
        servo_command = self._resolve_servo_command(event)
        if self.input_box.hasFocus() and not (servo_command is not None or self._is_global_control_key(key)):
            return False

        if event.isAutoRepeat():
            event.accept()
            return self._is_control_key(key) or servo_command is not None

        self._debug_key_press(event)

        if key == Qt.Key_Escape:
            self.input_box.clearFocus()
            self.setFocus(Qt.OtherFocusReason)
            event.accept()
            return True
        if key == Qt.Key_M:
            self.mic_btn.setChecked(not self.mic_btn.isChecked())
            self._toggle_mic()
            event.accept()
            return True
        if key == Qt.Key_T:
            self.request_handler("__TOGGLE_FOLLOW__", is_raw_command=True)
            event.accept()
            return True
        if key == Qt.Key_P:
            self.request_handler("__TOGGLE_AUTONOMOUS__", is_raw_command=True)
            event.accept()
            return True
        if key == Qt.Key_I:
            self.request_handler("__INSPECT_SCENE__", is_raw_command=True)
            event.accept()
            return True
        if key == Qt.Key_Space:
            self.request_handler("__E_STOP__", is_raw_command=True)
            self._last_drive_command = "S"
            event.accept()
            return True

        if key in self.DRIVE_KEY_MAP:
            self._pressed_drive_keys.add(key)
            self._key_counter += 1
            self._key_timestamps[key] = self._key_counter
            self.request_handler(self.DRIVE_KEY_MAP[key], is_raw_command=True)
            self._last_drive_command = self.DRIVE_KEY_MAP[key]
            event.accept()
            return True

        if servo_command is not None:
            self._pressed_servo_keys.add(servo_command)
            self._key_counter += 1
            self._key_timestamps[servo_command] = self._key_counter
            self.request_handler(servo_command, is_raw_command=True)
            event.accept()
            return True
        return False

    def _handle_key_release(self, event) -> bool:
        key = event.key()
        servo_command = self._resolve_servo_command(event)
        if event.isAutoRepeat():
            event.accept()
            return self._is_control_key(key) or servo_command is not None

        self._pressed_drive_keys.discard(key)
        if servo_command is not None:
            self._pressed_servo_keys.discard(servo_command)
        if key in self.DRIVE_KEY_MAP and not self._pressed_drive_keys:
            self.request_handler("S", is_raw_command=True)
            self._last_drive_command = "S"
            event.accept()
            return True
        if servo_command is not None:
            event.accept()
            return True
        return False

    def _is_control_key(self, key: int) -> bool:
        return (
            key in self.DRIVE_KEY_MAP
            or key in self.SERVO_KEY_MAP
            or key in {Qt.Key_Space, Qt.Key_T, Qt.Key_P, Qt.Key_I, Qt.Key_M, Qt.Key_Escape}
        )

    def _most_recent_servo_command(self) -> str | None:
        if not self._pressed_servo_keys:
            return None
        return max(self._pressed_servo_keys, key=lambda item: self._key_timestamps.get(item, 0))

    def _resolve_servo_command(self, event) -> str | None:
        key = event.key()
        native_virtual = 0
        try:
            native_virtual = int(event.nativeVirtualKey())
        except Exception:
            native_virtual = 0
        if native_virtual in self.NATIVE_SERVO_KEY_MAP:
            return self.NATIVE_SERVO_KEY_MAP[native_virtual]
        native_scan = 0
        try:
            native_scan = int(event.nativeScanCode())
        except Exception:
            native_scan = 0
        if native_scan in self.NATIVE_SCAN_SERVO_KEY_MAP:
            return self.NATIVE_SCAN_SERVO_KEY_MAP[native_scan]
        return self.SERVO_KEY_MAP.get(key)

    def _debug_key_press(self, event) -> None:
        now = time.monotonic()
        if (now - self._last_key_debug_at) < 0.05:
            return
        self._last_key_debug_at = now
        try:
            native_virtual = int(event.nativeVirtualKey())
        except Exception:
            native_virtual = 0
        try:
            native_scan = int(event.nativeScanCode())
        except Exception:
            native_scan = 0
        servo_command = self._resolve_servo_command(event) or "-"
        bus.emit(
            SystemEvents.LOG_MESSAGE,
            f"[DEBUG KEYCODE] Qt={int(event.key())} NativeVK={native_virtual} Scan={native_scan} Servo={servo_command}",
        )
