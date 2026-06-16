from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass, field
from typing import Optional

from config import RoverConfig
from core.event_bus import SystemEvents, bus
from modules.kalman_filter import Kalman2D, KalmanPoint
from modules.motor_controller import MotorController
from modules.pid_controller import PIDController
from modules.rover_control import RoverController
from modules.rover_types import TrackedTarget
from modules.servo_controller import ServoController


class MovingAverage2D:
    """Small fixed-window average for noisy detection centers."""

    def __init__(self, window_size: int) -> None:
        self._window_size = max(1, int(window_size))
        self._samples: deque[tuple[float, float]] = deque()
        self._sum_x = 0.0
        self._sum_y = 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0

    def update(self, x: float, y: float) -> tuple[float, float]:
        if len(self._samples) >= self._window_size:
            old_x, old_y = self._samples.popleft()
            self._sum_x -= old_x
            self._sum_y -= old_y
        sample = (float(x), float(y))
        self._samples.append(sample)
        self._sum_x += sample[0]
        self._sum_y += sample[1]
        count = max(1, len(self._samples))
        return self._sum_x / count, self._sum_y / count


@dataclass(slots=True)
class TrackingState:
    pan_angle: float = 90.0
    tilt_angle: float = 90.0
    last_detection_time: float = field(default_factory=time.monotonic)
    last_update_time: float = field(default_factory=time.monotonic)
    target_locked: bool = False
    last_drive_command: str | None = None
    last_servo_command: tuple[int, int] | None = None
    prediction_frames: int = 0
    predicted_point: tuple[int, int] | None = None
    smoothed_target_point: tuple[float, float] | None = None
    last_tracking_status: str = "IDLE"
    active_target_id: int | None = None
    last_pan_delta: float = 0.0
    last_tilt_delta: float = 0.0
    last_seen_servo_pose: tuple[int, int] | None = None


class TrackingController:
    """Drive and pan / tilt follow logic for the active tracked target."""

    def __init__(
        self,
        config: RoverConfig,
        rover: RoverController,
        servo: ServoController,
        motor: MotorController,
    ) -> None:
        self._config = config
        self._rover = rover
        self._servo = servo
        self._motor = motor
        self._state = TrackingState()
        self._state.pan_angle = float(config.servo_center_angle)
        self._state.tilt_angle = float(config.servo_center_angle)
        self._kalman = Kalman2D(
            process_noise=config.kalman_process_noise,
            measurement_noise=config.kalman_measurement_noise,
        )
        self._measurement_filter = MovingAverage2D(config.tracking_moving_average_window)
        self._pan_pid = PIDController(
            kp=config.pan_pid_kp,
            ki=config.pan_pid_ki,
            kd=config.pan_pid_kd,
            integral_limit=config.pid_integral_limit,
            output_limit=config.servo_max_step_deg,
        )
        self._tilt_pid = PIDController(
            kp=config.tilt_pid_kp,
            ki=config.tilt_pid_ki,
            kd=config.tilt_pid_kd,
            integral_limit=config.pid_integral_limit,
            output_limit=config.servo_max_step_deg,
        )

    def update(self, target: Optional[TrackedTarget], frame_w: int, frame_h: int) -> str:
        try:
            self._state.last_tracking_status = self.update_servos(target, frame_w, frame_h)
            if target is None:
                return self._dispatch_drive("S")
            return self._dispatch_drive(self._drive_command(target, frame_w, frame_h))
        except Exception as exc:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[TrackingController] update error: {exc}")
            return "ERROR"

    def update_servos(self, target: Optional[TrackedTarget], frame_w: int, frame_h: int) -> str:
        """Aim pan/tilt at the locked target without issuing motor drive commands."""
        now = time.monotonic()
        nominal_dt = 1.0 / max(1, self._config.servo_send_hz)
        dt = max(nominal_dt, now - self._state.last_update_time)
        self._state.last_update_time = now

        if target is None:
            if self._bridge_recent_detection_gap(dt):
                return self._state.last_tracking_status
            if bool(getattr(self._config, "tracking_predict_on_loss", False)):
                predicted = self._predict_target_center(dt, frame_w, frame_h)
                if predicted is not None:
                    self._apply_servo_to_point(predicted.x, predicted.y, frame_w, frame_h, dt)
                    self._state.target_locked = True
                    self._state.last_tracking_status = "PREDICT"
                    return "PREDICT"
            self.hold_last_seen_pose()
            return self._state.last_tracking_status

        if self._state.active_target_id != target.target_id:
            self._start_target_session(target.target_id)

        self._state.last_detection_time = now
        measured_x, measured_y = self._smooth_measurement(target.bbox.center_x, target.bbox.center_y)
        estimate = self._kalman.update(measured_x, measured_y, dt)
        self._state.prediction_frames = 0
        latency = max(0.0, float(getattr(self._config, "servo_hardware_latency_seconds", 0.0)))
        aim_point = self._kalman.project(latency) or estimate
        self._state.predicted_point = self._clamp_point(aim_point.x, aim_point.y, frame_w, frame_h)
        offset_x, offset_y = self._compute_offsets_from_point(aim_point.x, aim_point.y, frame_w, frame_h)
        self._state.target_locked = self._within_dead_zone(offset_x, offset_y) and (
            target.stable_frames >= self._config.target_lock_frames
        )

        self._apply_servo_to_point(aim_point.x, aim_point.y, frame_w, frame_h, dt)
        self._state.last_seen_servo_pose = self.current_angles()
        self._state.last_tracking_status = "TRACK"
        return "TRACK"

    def manual_pan_tilt(self, *, pan_delta: float = 0, tilt_delta: float = 0) -> tuple[int, int]:
        self._state.pan_angle = self._clamp_pan_angle(self._state.pan_angle + pan_delta)
        self._state.tilt_angle = self._clamp_tilt_angle(self._state.tilt_angle + tilt_delta)
        self._state.target_locked = False
        self._state.last_seen_servo_pose = self.current_angles()
        self._pan_pid.reset()
        self._tilt_pid.reset()
        pan = int(round(self._state.pan_angle))
        tilt = int(round(self._state.tilt_angle))
        if pan_delta and not tilt_delta:
            self._dispatch_servo_axis("Pan", pan)
        elif tilt_delta and not pan_delta:
            self._dispatch_servo_axis("Tilt", tilt)
        else:
            self._dispatch_servo(pan, tilt)
        return pan, tilt

    def reset(self) -> None:
        try:
            self.clear_lock_state()
            center = float(self._config.servo_center_angle)
            pan_center = int(self._clamp_pan_angle(center))
            tilt_center = int(self._clamp_tilt_angle(center))
            self._state.pan_angle = float(pan_center)
            self._state.tilt_angle = float(tilt_center)
            self._state.last_seen_servo_pose = (pan_center, tilt_center)
            self._dispatch_drive("S")
            self._dispatch_servo(pan_center, tilt_center)
        except Exception as exc:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[TrackingController] reset error: {exc}")

    def clear_lock_state(self) -> None:
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._kalman.reset()
        self._measurement_filter.reset()
        self._state.target_locked = False
        self._state.prediction_frames = 0
        self._state.predicted_point = None
        self._state.smoothed_target_point = None
        self._state.last_tracking_status = "IDLE"
        self._state.active_target_id = None
        self._state.last_pan_delta = 0.0
        self._state.last_tilt_delta = 0.0

    def hold_last_seen_pose(self) -> tuple[int, int]:
        """Stop predictive motion and keep the camera at the last confirmed target pose."""
        pose = self._state.last_seen_servo_pose or self.current_angles()
        pan = int(round(self._clamp_pan_angle(float(pose[0]))))
        tilt = int(round(self._clamp_tilt_angle(float(pose[1]))))
        self._state.pan_angle = float(pan)
        self._state.tilt_angle = float(tilt)
        self.clear_lock_state()
        self._state.last_seen_servo_pose = (pan, tilt)
        self._state.last_tracking_status = "SEARCH"
        self._dispatch_servo(pan, tilt)
        return pan, tilt

    def current_angles(self) -> tuple[int, int]:
        return int(round(self._state.pan_angle)), int(round(self._state.tilt_angle))

    def target_locked(self) -> bool:
        return self._state.target_locked

    def latency_ms(self) -> float:
        return self._servo.latency_ms()

    def predicted_point(self) -> tuple[int, int] | None:
        return self._state.predicted_point

    def prediction_path(self) -> tuple[tuple[int, int], ...]:
        return self._kalman.history()

    def tracking_status(self) -> str:
        return self._state.last_tracking_status

    def _dispatch_drive(self, command: str) -> str:
        if command != self._state.last_drive_command:
            self._rover.send_command(command)
            self._motor.send(command)
            self._state.last_drive_command = command
        return command

    def stop_drive(self) -> str:
        return self._dispatch_drive("S")

    def hard_stop_drive(self) -> str:
        self._rover.send_command("S")
        force_stop = getattr(self._motor, "force_stop", None)
        if callable(force_stop):
            force_stop()
        else:
            self._motor.send("S")
        self._state.last_drive_command = "S"
        return "S"

    def _dispatch_servo(self, pan: int, tilt: int) -> None:
        command = (pan, tilt)
        if command != self._state.last_servo_command:
            self._servo.send_pan_tilt(pan, tilt)
            self._state.last_servo_command = command

    def _dispatch_servo_axis(self, axis: str, value: int) -> None:
        pan, tilt = self._state.last_servo_command or self.current_angles()
        if axis == "Pan":
            command = (value, tilt)
            payload = f"Pan,{value}"
        else:
            command = (pan, value)
            payload = f"Tilt,{value}"
        if command != self._state.last_servo_command:
            self._servo.send(payload)
            self._state.last_servo_command = command

    def _predict_target_center(self, dt: float, frame_w: int, frame_h: int) -> KalmanPoint | None:
        if not self._kalman.active():
            return None
        if self._state.prediction_frames >= self._config.kalman_max_prediction_frames:
            self._kalman.reset()
            self._state.prediction_frames = 0
            return None
        predicted = self._kalman.predict(dt)
        if predicted is None:
            return None
        self._state.prediction_frames += 1
        clamped_x, clamped_y = self._clamp_point(predicted.x, predicted.y, frame_w, frame_h)
        self._state.predicted_point = (clamped_x, clamped_y)
        predicted.x = clamped_x
        predicted.y = clamped_y
        return predicted

    def _apply_servo_to_point(self, x: float, y: float, frame_w: int, frame_h: int, dt: float) -> None:
        offset_x, offset_y = self._compute_offsets_from_point(x, y, frame_w, frame_h)
        deadband = self._servo_deadband_px()
        error_x = self._normalize_error(offset_x, frame_w)
        error_y = self._normalize_error(offset_y, frame_h)
        if abs(offset_x) <= deadband:
            error_x = 0.0
            self._state.last_pan_delta = 0.0
            self._pan_pid.reset()
        if abs(offset_y) <= deadband:
            error_y = 0.0
            self._state.last_tilt_delta = 0.0
            self._tilt_pid.reset()
        if error_x == 0.0 and error_y == 0.0:
            return

        pan_direction = 1.0 if self._config.servo_tracking_pan_direction >= 0 else -1.0
        tilt_direction = 1.0 if self._config.servo_tracking_tilt_direction >= 0 else -1.0
        pan_delta = 0.0
        tilt_delta = 0.0
        if error_x != 0.0:
            pan_delta = (
                pan_direction
                * self._pan_pid.update(error_x, dt)
                * self._axis_ease(abs(offset_x), frame_w, deadband)
            )
        if error_y != 0.0:
            tilt_delta = (
                tilt_direction
                * self._tilt_pid.update(error_y, dt)
                * self._axis_ease(abs(offset_y), frame_h, deadband)
            )
        max_delta = max(0.0, float(self._config.servo_max_speed_deg_per_sec)) * max(1e-3, dt)
        if max_delta > 0.0:
            pan_delta = max(-max_delta, min(max_delta, pan_delta))
            tilt_delta = max(-max_delta, min(max_delta, tilt_delta))
        min_delta = max(0.0, float(self._config.servo_min_delta_deg))
        if abs(pan_delta) < min_delta:
            pan_delta = 0.0
        if abs(tilt_delta) < min_delta:
            tilt_delta = 0.0
        pan_delta = self._smooth_axis_delta(pan_delta, self._state.last_pan_delta)
        tilt_delta = self._smooth_axis_delta(tilt_delta, self._state.last_tilt_delta)
        if abs(pan_delta) < min_delta:
            pan_delta = 0.0
        if abs(tilt_delta) < min_delta:
            tilt_delta = 0.0
        self._state.last_pan_delta = pan_delta
        self._state.last_tilt_delta = tilt_delta
        if pan_delta == 0.0 and tilt_delta == 0.0:
            return
        self._state.pan_angle = self._clamp_pan_angle(self._state.pan_angle + pan_delta)
        self._state.tilt_angle = self._clamp_tilt_angle(self._state.tilt_angle + tilt_delta)
        self._dispatch_servo(int(round(self._state.pan_angle)), int(round(self._state.tilt_angle)))

    def _start_target_session(self, target_id: int) -> None:
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._kalman.reset()
        self._state.prediction_frames = 0
        self._state.predicted_point = None
        self._state.smoothed_target_point = None
        self._state.target_locked = False
        self._state.active_target_id = target_id
        self._state.last_pan_delta = 0.0
        self._state.last_tilt_delta = 0.0
        self._measurement_filter.reset()

    def _smooth_axis_delta(self, delta: float, previous_delta: float) -> float:
        if delta == 0.0:
            return 0.0
        alpha = max(0.05, min(1.0, float(getattr(self._config, "servo_motion_smoothing_alpha", 1.0))))
        if previous_delta and ((delta > 0) != (previous_delta > 0)):
            previous_delta = 0.0
        if previous_delta == 0.0:
            return delta
        return (alpha * delta) + ((1.0 - alpha) * previous_delta)

    def _bridge_recent_detection_gap(self, dt: float) -> bool:
        """Coast briefly through one network/inference hiccup, then yield to safe hold."""
        bridge_seconds = max(0.0, float(getattr(self._config, "tracking_loss_bridge_seconds", 0.0)))
        if bridge_seconds <= 0.0:
            return False
        gap = time.monotonic() - self._state.last_detection_time
        if gap > bridge_seconds:
            return False
        if self._state.last_seen_servo_pose is None:
            return False

        scale = max(0.0, min(1.0, float(getattr(self._config, "tracking_loss_bridge_velocity_scale", 0.5))))
        pan_delta = self._state.last_pan_delta * scale
        tilt_delta = self._state.last_tilt_delta * scale
        max_delta = max(0.0, float(self._config.servo_max_speed_deg_per_sec)) * max(1e-3, dt)
        if max_delta > 0.0:
            pan_delta = max(-max_delta, min(max_delta, pan_delta))
            tilt_delta = max(-max_delta, min(max_delta, tilt_delta))

        min_delta = max(0.25, float(self._config.servo_min_delta_deg) * 0.5)
        if abs(pan_delta) < min_delta:
            pan_delta = 0.0
        if abs(tilt_delta) < min_delta:
            tilt_delta = 0.0
        if pan_delta == 0.0 and tilt_delta == 0.0:
            return False

        self._state.pan_angle = self._clamp_pan_angle(self._state.pan_angle + pan_delta)
        self._state.tilt_angle = self._clamp_tilt_angle(self._state.tilt_angle + tilt_delta)
        self._state.last_pan_delta = pan_delta
        self._state.last_tilt_delta = tilt_delta
        self._state.target_locked = False
        self._state.predicted_point = None
        self._state.last_tracking_status = "BRIDGE"
        self._dispatch_servo(int(round(self._state.pan_angle)), int(round(self._state.tilt_angle)))
        return True

    def _axis_ease(self, abs_offset: float, frame_span: int, deadband: int) -> float:
        active_span = max(1.0, (frame_span / 2.0) - float(deadband))
        normalized = max(0.0, min(1.0, (float(abs_offset) - float(deadband)) / active_span))
        min_ease = max(0.05, min(1.0, float(self._config.servo_easing_min)))
        exponent = max(0.1, float(self._config.servo_easing_exponent))
        return min_ease + ((1.0 - min_ease) * (normalized**exponent))

    def _smooth_measurement(self, x: float, y: float) -> tuple[float, float]:
        x, y = self._measurement_filter.update(x, y)
        alpha = max(0.0, min(1.0, float(self._config.tracking_measurement_alpha)))
        previous = self._state.smoothed_target_point
        if previous is None or alpha >= 1.0:
            smoothed = (float(x), float(y))
        else:
            smoothed = (
                (alpha * float(x)) + ((1.0 - alpha) * previous[0]),
                (alpha * float(y)) + ((1.0 - alpha) * previous[1]),
            )
        self._state.smoothed_target_point = smoothed
        return smoothed

    def _clamp_pan_angle(self, angle: float) -> float:
        return max(float(self._config.servo_pan_min_angle), min(float(self._config.servo_pan_max_angle), angle))

    def _clamp_tilt_angle(self, angle: float) -> float:
        return max(float(self._config.servo_tilt_min_angle), min(float(self._config.servo_tilt_max_angle), angle))

    @staticmethod
    def _normalize_error(offset: float, frame_span: int) -> float:
        half = max(1.0, frame_span / 2.0)
        return offset / half

    def _within_dead_zone(self, offset_x: float, offset_y: float) -> bool:
        deadband = self._servo_deadband_px()
        return abs(offset_x) <= deadband and abs(offset_y) <= deadband

    def _servo_deadband_px(self) -> int:
        return max(1, int(getattr(self._config, "tracking_deadband_px", self._config.dead_zone_px)))

    def _compute_offsets(
        self,
        target: TrackedTarget,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        offset_x = target.bbox.center_x - frame_w / 2
        offset_y = target.bbox.center_y - frame_h / 2
        return offset_x, offset_y

    @staticmethod
    def _compute_offsets_from_point(
        x: float,
        y: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        offset_x = x - frame_w / 2
        offset_y = y - frame_h / 2
        return offset_x, offset_y

    @staticmethod
    def _clamp_point(x: float, y: float, frame_w: int, frame_h: int) -> tuple[int, int]:
        max_x = max(0, frame_w - 1)
        max_y = max(0, frame_h - 1)
        return (
            max(0, min(max_x, int(round(x)))),
            max(0, min(max_y, int(round(y)))),
        )

    def _drive_command(self, target: TrackedTarget, frame_w: int, frame_h: int) -> str:
        if target.stable_frames < self._config.target_lock_frames:
            return "S"
        pan_bias = self._pan_alignment_bias()
        threshold = max(1.0, float(self._config.follow_pan_align_threshold_deg))
        if pan_bias >= threshold:
            return "L"
        if pan_bias <= -threshold:
            return "R"
        if not self._state.target_locked:
            return "S"
        fraction = target.bbox.area / (frame_w * frame_h)
        if fraction < self._config.bbox_min_fraction:
            return "F"
        if fraction > self._config.bbox_max_fraction:
            return "B"
        return "S"

    def _pan_alignment_bias(self) -> float:
        center = float(self._config.servo_center_angle)
        pan_error = self._state.pan_angle - center
        pan_direction = 1.0 if self._config.servo_tracking_pan_direction >= 0 else -1.0
        return pan_error * pan_direction
