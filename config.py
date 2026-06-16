from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

# Try to load python-dotenv if available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_DIR = PROJECT_ROOT / "local_config"
LOCAL_ENV_FILE = LOCAL_CONFIG_DIR / ".env"

try:
    load_dotenv(LOCAL_ENV_FILE)
except NameError:
    pass

DEFAULT_KNOWLEDGE_PATHS = (
    "README.md",
    "config.py",
    "modules",
)
DEFAULT_TRACKER_CONFIG = "trackers/rover_botsort.yaml"
DEFAULT_ESP32_IP = os.getenv("ROVER_ESP32_IP", "192.168.0.101")
DEFAULT_CAMERA_IP = os.getenv("ROVER_CAMERA_IP", "192.168.0.100")
DEFAULT_CAMERA_STREAM_URL = os.getenv("ROVER_CAMERA_STREAM_URL", f"http://{DEFAULT_CAMERA_IP}:81/stream")
DEFAULT_JARVIS_WS_URL = f"ws://{DEFAULT_ESP32_IP}:80/Jarvis"
DEFAULT_DEV_BOARD_UDP_PORT = int(os.getenv("ROVER_DEV_BOARD_UDP_PORT", "4210"))
DEFAULT_DEV_BOARD_UDP_URL = os.getenv("ROVER_DEV_BOARD_UDP_URL", f"udp://{DEFAULT_ESP32_IP}:{DEFAULT_DEV_BOARD_UDP_PORT}")
DEFAULT_SERVO_WS_URL = os.getenv("ROVER_SERVO_WS_URL", DEFAULT_JARVIS_WS_URL)
DEFAULT_MOTOR_WS_URL = os.getenv("ROVER_MOTOR_WS_URL", DEFAULT_JARVIS_WS_URL)
DEFAULT_SERVO_URL = os.getenv("ROVER_SERVO_URL", DEFAULT_DEV_BOARD_UDP_URL)
DEFAULT_MOTOR_URL = os.getenv("ROVER_MOTOR_URL", DEFAULT_DEV_BOARD_UDP_URL)
DEFAULT_TRANSPORT_PROTOCOL = (os.getenv("ROVER_TRANSPORT_PROTOCOL", "legacy_csv") or "legacy_csv").strip().lower()
DEFAULT_STATUS_LED_COLOR = os.getenv("ROVER_STATUS_LED", "blue")
DEFAULT_PERFORMANCE_PROFILE = (os.getenv("VISION_PERF_PROFILE", "rtx5060") or "rtx5060").strip().lower()


class Config:
    API_TIMEOUT = 20

    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen2.5vl:3b")

    SINGLE_INSTANCE_SERVER = "VISIONControlPanel"
    LAUNCHER_APP_NAME = "V.I.S.I.O.N Launcher"
    MAIN_WINDOW_TITLE = "V.I.S.I.O.N Control Panel"


@dataclass(slots=True)
class RoverConfig:
    # Network
    vision_stream_url: str
    servo_url: str
    motor_url: str

    # UI / timing
    performance_profile: str = "rtx5060"
    vision_loop_hz: int = 30
    ui_frame_hz: int = 30
    detection_hz: int = 8
    snapshot_poll_hz: int = 12
    key_repeat_hz: int = 30
    frame_stale_seconds: float = 2.5
    camera_disconnect_timeout: float = 1.0
    camera_initial_frame_timeout: float = 8.0
    camera_reconnect_timeout: float = 12.0
    camera_http_read_size: int = 16_384
    camera_enable_ffmpeg_fallback: bool = False
    camera_flip_code: int | None = -1

    # Manual pan / tilt
    servo_step: float = 6.0
    servo_center_angle: int = 90
    servo_min_angle: int = 10
    servo_max_angle: int = 170
    servo_pan_min_angle: int = 10
    servo_pan_max_angle: int = 170
    servo_tilt_min_angle: int = 10
    servo_tilt_max_angle: int = 155
    servo_manual_pan_direction: int = -1
    servo_manual_tilt_direction: int = 1
    servo_tracking_pan_direction: int = 1
    servo_tracking_tilt_direction: int = 1
    servo_send_hz: int = 30
    servo_max_step_deg: float = 4.0
    servo_max_speed_deg_per_sec: float = 96.0
    servo_min_delta_deg: float = 0.40
    servo_motion_smoothing_alpha: float = 0.42
    servo_easing_min: float = 0.24
    servo_easing_exponent: float = 1.28
    motor_drive_speed: int = 170
    motor_turn_speed: int = 150
    motor_send_hz: int = 20
    motor_command_ttl_seconds: float = 0.35

    # Follow-person tracking
    dead_zone_px: int = 30
    pan_tilt_gain: float = 0.05
    bbox_min_fraction: float = 0.10
    bbox_max_fraction: float = 0.30
    no_detection_timeout: float = 2.0
    max_target_lost_frames: int = 30
    track_iou_threshold: float = 0.24
    duplicate_detection_iou_threshold: float = 0.72
    target_acquisition_frames: int = 1
    target_rebind_frames: int = 2
    target_lock_frames: int = 3
    target_center_weight: float = 0.65
    target_front_area_weight: float = 2.4
    target_min_acquire_area_fraction: float = 0.0025
    target_box_smoothing_alpha: float = 0.18
    follow_pan_align_threshold_deg: float = 12.0
    scene_announce_cooldown_seconds: float = 6.0
    face_lock_enabled: bool = True
    face_lock_yolo_fallback_enabled: bool = True
    face_detector_scale_factor: float = 1.08
    face_detector_min_neighbors: int = 3
    face_detector_min_size_px: int = 18
    face_detector_max_faces: int = 3
    face_proxy_width_fraction: float = 0.42
    face_proxy_height_fraction: float = 0.24
    face_proxy_y_fraction: float = 0.07
    tracking_moving_average_window: int = 3
    # PID constants are intentionally damped for smooth deceleration near center.
    pid_integral_limit: float = 1.6
    pan_pid_kp: float = 13.0
    pan_pid_ki: float = 0.55
    pan_pid_kd: float = 2.8
    tilt_pid_kp: float = 11.0
    tilt_pid_ki: float = 0.45
    tilt_pid_kd: float = 2.3
    tracking_deadband_px: int = 16
    tracking_measurement_alpha: float = 0.24
    tracking_predict_on_loss: bool = False
    tracking_loss_bridge_seconds: float = 0.09
    tracking_loss_bridge_velocity_scale: float = 0.5
    kalman_max_prediction_frames: int = 30
    kalman_process_noise: float = 35.0
    kalman_measurement_noise: float = 90.0
    servo_hardware_latency_seconds: float = 0.08

    # Autonomous navigation
    autonomous_stop_fraction: float = 0.26
    autonomous_turn_fraction: float = 0.12
    autonomous_lane_margin: float = 0.025
    autonomous_turn_hold_seconds: float = 0.55
    autonomous_min_detection_fraction: float = 0.012
    autonomous_clear_frames_required: int = 2

    # Transport
    ws_recv_timeout: float = 2.0
    reconnect_interval: float = 3.0
    transport_protocol: str = DEFAULT_TRANSPORT_PROTOCOL
    status_led_color: str = DEFAULT_STATUS_LED_COLOR
    dev_board_udp_port: int = DEFAULT_DEV_BOARD_UDP_PORT

    # Detector
    detector_backend: str = "yolo26"
    detector_model: str = "yolo26n.pt"
    detector_fallback_model: str = "yolov8n.pt"
    detector_confidence: float = 0.30
    detector_tracking_confidence: float = 0.30
    detector_tracking_iou: float = 0.55
    detector_device: str = "auto"
    detector_half_precision: bool = True
    detector_max_detections: int = 4
    detector_input_width: int = 416
    detector_track_classes: tuple[int, ...] | None = (0,)
    detector_tracker_config: str = DEFAULT_TRACKER_CONFIG
    target_label: str = "person"

    # Audio
    audio_sample_rate: int = 16_000
    audio_chunk_size: int = 1_024
    speech_activation_threshold: float = 0.02
    speech_silence_seconds: float = 0.90
    clap_amplitude_threshold: float = 0.12
    clap_min_separation_seconds: float = 0.16
    clap_window_seconds: float = 0.85
    clap_cooldown_seconds: float = 2.0

    # STT / TTS
    stt_backend: str = "faster_whisper"
    stt_model_size: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "en"
    tts_backend: str = "offline"
    tts_voice_hint: str = "firm"

    # Local knowledge + scene reasoning
    knowledge_paths: tuple[str, ...] = field(default_factory=lambda: DEFAULT_KNOWLEDGE_PATHS)
    vlm_endpoint: str | None = None
    vlm_model: str | None = None

    # App / launcher
    single_instance_server: str = Config.SINGLE_INSTANCE_SERVER
    main_window_title: str = Config.MAIN_WINDOW_TITLE
    launcher_name: str = Config.LAUNCHER_APP_NAME

    @property
    def yolo_model(self) -> str:
        return self.detector_model

    @property
    def yolo_confidence(self) -> float:
        return self.detector_confidence

    @property
    def resolved_knowledge_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for value in self.knowledge_paths:
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths.append(path)
        return tuple(paths)

    @property
    def resolved_tracker_config_path(self) -> Path:
        path = Path(self.detector_tracker_config)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


PERFORMANCE_PROFILES: dict[str, dict[str, int | float | str]] = {
    "mx330": {
        "performance_profile": "mx330",
        "vision_loop_hz": 30,
        "ui_frame_hz": 30,
        "detection_hz": 8,
        "snapshot_poll_hz": 12,
        "detector_input_width": 416,
    },
    "rtx5060": {
        "performance_profile": "rtx5060",
        "vision_loop_hz": 60,
        "ui_frame_hz": 60,
        "detection_hz": 30,
        "snapshot_poll_hz": 30,
        "detector_input_width": 960,
        "detector_device": "cuda:0",
        "detector_confidence": 0.32,
        "detector_tracking_confidence": 0.36,
        "detector_tracking_iou": 0.60,
        "servo_step": 6.0,
        "servo_send_hz": 36,
        "servo_max_step_deg": 4.8,
        "servo_max_speed_deg_per_sec": 128.0,
        "servo_min_delta_deg": 0.32,
        "servo_motion_smoothing_alpha": 0.48,
        "servo_easing_min": 0.26,
        "servo_easing_exponent": 1.18,
        "target_acquisition_frames": 1,
        "target_rebind_frames": 1,
        "target_box_smoothing_alpha": 0.16,
        "pan_pid_kp": 12.0,
        "pan_pid_ki": 0.45,
        "pan_pid_kd": 2.6,
        "tilt_pid_kp": 10.0,
        "tilt_pid_ki": 0.35,
        "tilt_pid_kd": 2.1,
        "tracking_deadband_px": 20,
        "tracking_measurement_alpha": 0.18,
        "tracking_moving_average_window": 3,
        "tracking_loss_bridge_seconds": 0.08,
        "tracking_loss_bridge_velocity_scale": 0.5,
        "servo_hardware_latency_seconds": 0.08,
        "kalman_max_prediction_frames": 30,
        "kalman_process_noise": 18.0,
        "kalman_measurement_noise": 130.0,
    },
}


def build_rover_config(profile: str | None = None) -> RoverConfig:
    selected = (profile or DEFAULT_PERFORMANCE_PROFILE or "mx330").strip().lower()
    if selected not in PERFORMANCE_PROFILES:
        selected = "mx330"
    overrides = PERFORMANCE_PROFILES[selected]
    return RoverConfig(
        vision_stream_url=DEFAULT_CAMERA_STREAM_URL,
        servo_url=DEFAULT_SERVO_URL,
        motor_url=DEFAULT_MOTOR_URL,
        ws_recv_timeout=5.0,
        reconnect_interval=2.0,
        **overrides,
    )


rover_config = build_rover_config()
