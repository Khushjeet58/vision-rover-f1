from pathlib import Path

from config import (
    DEFAULT_CAMERA_STREAM_URL,
    DEFAULT_MOTOR_URL,
    DEFAULT_SERVO_URL,
    PROJECT_ROOT,
    RoverConfig,
    build_rover_config,
)


DUMMY_CONFIG = RoverConfig(
    vision_stream_url="ws://localhost/Camera",
    servo_url="udp://localhost:4210",
    motor_url="udp://localhost:4210",
)


def test_required_fields_exist():
    cfg = DUMMY_CONFIG
    assert cfg.detector_backend == "yolo26"
    assert cfg.target_label == "person"
    assert cfg.detector_half_precision is True
    assert cfg.detector_max_detections == 4
    assert cfg.detector_confidence == 0.30
    assert cfg.detector_track_classes == (0,)
    assert cfg.camera_flip_code == -1
    assert cfg.servo_step == 6.0
    assert cfg.servo_send_hz > 0
    assert cfg.servo_motion_smoothing_alpha > 0
    assert cfg.servo_easing_min > 0
    assert cfg.servo_easing_exponent > 0
    assert cfg.servo_center_angle == 90
    assert cfg.servo_min_angle == 10
    assert cfg.servo_max_angle == 170
    assert cfg.servo_pan_min_angle == 10
    assert cfg.servo_pan_max_angle == 170
    assert cfg.servo_tilt_min_angle == 10
    assert cfg.servo_tilt_max_angle == 155
    assert cfg.servo_manual_tilt_direction == 1
    assert cfg.servo_tracking_tilt_direction == 1
    assert cfg.servo_manual_pan_direction in {-1, 1}
    assert cfg.servo_manual_tilt_direction in {-1, 1}
    assert cfg.servo_tracking_pan_direction in {-1, 1}
    assert cfg.servo_tracking_tilt_direction in {-1, 1}
    assert cfg.motor_drive_speed > 0
    assert cfg.motor_turn_speed > 0
    assert cfg.motor_send_hz > 0
    assert cfg.motor_command_ttl_seconds > 0
    assert cfg.transport_protocol == "legacy_csv"
    assert cfg.status_led_color == "blue"
    assert cfg.autonomous_clear_frames_required >= 1
    assert cfg.pan_pid_kp > 0
    assert cfg.tilt_pid_kp > 0
    assert cfg.tracking_deadband_px > 0
    assert cfg.follow_pan_align_threshold_deg > 0
    assert cfg.face_lock_enabled is True
    assert cfg.face_lock_yolo_fallback_enabled is True
    assert cfg.face_detector_min_size_px > 0
    assert cfg.face_proxy_width_fraction > 0
    assert cfg.face_proxy_height_fraction > 0
    assert cfg.face_proxy_y_fraction >= 0
    assert cfg.kalman_max_prediction_frames == 30
    assert cfg.kalman_process_noise > 0
    assert cfg.kalman_measurement_noise > 0
    assert cfg.servo_hardware_latency_seconds >= 0
    assert cfg.detector_tracking_iou > 0
    assert cfg.target_acquisition_frames >= 1
    assert cfg.target_rebind_frames >= 1
    assert cfg.target_min_acquire_area_fraction > 0
    assert cfg.tracking_moving_average_window >= 1
    assert cfg.tracking_predict_on_loss is False
    assert cfg.tracking_loss_bridge_seconds > 0
    assert 0 < cfg.tracking_loss_bridge_velocity_scale <= 1
    assert cfg.resolved_tracker_config_path.name == "rover_botsort.yaml"
    assert cfg.audio_sample_rate > 0
    assert cfg.key_repeat_hz > 0
    assert cfg.stt_backend
    assert cfg.tts_backend


def test_url_fields_are_strings():
    assert DUMMY_CONFIG.vision_stream_url.startswith(("ws://", "wss://", "http://", "https://"))
    assert DUMMY_CONFIG.servo_url.startswith(("ws://", "udp://"))
    assert DUMMY_CONFIG.motor_url.startswith(("ws://", "udp://"))


def test_build_rover_config_defaults_to_shared_udp_dev_board_endpoint():
    cfg = build_rover_config("rtx5060")

    assert cfg.servo_url == DEFAULT_SERVO_URL
    assert cfg.motor_url == DEFAULT_MOTOR_URL
    assert cfg.servo_url == cfg.motor_url


def test_build_rover_config_uses_separate_default_camera_ip():
    cfg = build_rover_config("rtx5060")

    assert cfg.vision_stream_url == DEFAULT_CAMERA_STREAM_URL
    assert cfg.servo_url == DEFAULT_SERVO_URL


def test_legacy_yolo_properties_map_to_detector_fields():
    assert DUMMY_CONFIG.yolo_model == DUMMY_CONFIG.detector_model
    assert DUMMY_CONFIG.yolo_confidence == DUMMY_CONFIG.detector_confidence


def test_bbox_thresholds_are_ordered():
    assert DUMMY_CONFIG.bbox_min_fraction < DUMMY_CONFIG.bbox_max_fraction


def test_knowledge_paths_resolve_relative_to_project_root():
    resolved = DUMMY_CONFIG.resolved_knowledge_paths
    assert all(isinstance(path, Path) for path in resolved)
    assert all(path.is_absolute() for path in resolved)
    assert resolved[0].parent == PROJECT_ROOT


def test_tracker_config_resolves_relative_to_project_root():
    resolved = DUMMY_CONFIG.resolved_tracker_config_path
    assert resolved.is_absolute()
    assert resolved.parent == PROJECT_ROOT / "trackers"


def test_build_rover_config_uses_mx330_profile_defaults():
    cfg = build_rover_config("mx330")

    assert cfg.performance_profile == "mx330"
    assert cfg.vision_loop_hz == 30
    assert cfg.ui_frame_hz == 30
    assert cfg.detection_hz == 8
    assert cfg.snapshot_poll_hz == 12
    assert cfg.detector_input_width == 416


def test_build_rover_config_uses_rtx5060_profile_defaults():
    cfg = build_rover_config("rtx5060")

    assert cfg.performance_profile == "rtx5060"
    assert cfg.vision_loop_hz == 60
    assert cfg.ui_frame_hz == 60
    assert cfg.detection_hz == 30
    assert cfg.snapshot_poll_hz == 30
    assert cfg.detector_input_width == 960
    assert cfg.detector_device == "cuda:0"
    assert cfg.detector_confidence == 0.32
    assert cfg.detector_tracking_confidence == 0.36
    assert cfg.servo_step == 6.0
    assert cfg.servo_send_hz == 36
    assert cfg.target_acquisition_frames == 1
    assert cfg.tracking_deadband_px == 20
    assert cfg.tracking_moving_average_window == 3


def test_build_rover_config_falls_back_to_mx330_for_unknown_profile():
    cfg = build_rover_config("mystery-gpu")

    assert cfg.performance_profile == "mx330"
    assert cfg.vision_loop_hz == 30
    assert cfg.ui_frame_hz == 30
