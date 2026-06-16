from __future__ import annotations

from collections import Counter
import re

from modules.command_handler import CommandHandler
from modules.knowledge_base import KnowledgeBase
from modules.rover_types import ConnectionState, ControlMode, VisionSnapshot
from modules.scene_perception import ScenePerceptionService


class OperatorAssistant:
    """Fast local answers for common rover questions."""

    _WS_RE = re.compile(r"\s+")

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self._scene_perception = ScenePerceptionService()
        self._knowledge_base = knowledge_base

    def try_answer(self, text: str, snapshot: VisionSnapshot) -> str | None:
        query = (text or "").strip().lower()
        if not query:
            return None

        if self._is_help_question(query):
            return self._command_summary()
        if self._is_project_question(query):
            return self._project_summary(query)
        if self._is_scene_question(query):
            return self._scene_summary(snapshot)
        if self._is_status_question(query):
            return self._status_summary(snapshot)
        if self._is_connection_question(query):
            return self._connection_summary(snapshot)
        if self._is_target_question(query):
            return self._target_summary(snapshot)
        if self._is_capability_question(query):
            return (
                "I can answer basic rover questions, inspect the live scene, drive manually, "
                "follow one person on command, and run autonomous navigation from the camera feed."
            )
        return None

    def build_runtime_context(self, snapshot: VisionSnapshot) -> str:
        detections = list(snapshot.detections or [])
        counts = Counter(det.label.lower() for det in detections if det.label)
        if counts:
            detection_text = ", ".join(f"{label}:{count}" for label, count in sorted(counts.items()))
        else:
            detection_text = "none"

        if snapshot.target is not None:
            target_text = (
                f"locked {snapshot.target.label} at {int(snapshot.target.bbox.center_x)},"
                f"{int(snapshot.target.bbox.center_y)}"
            )
        else:
            target_text = "no active target lock"

        connection_parts = []
        for channel, state in sorted((snapshot.links or {}).items()):
            state_text = state.value.lower() if isinstance(state, ConnectionState) else str(state).lower()
            connection_parts.append(f"{channel}:{state_text}")
        if not connection_parts:
            connection_parts.append("no link telemetry")

        return (
            f"mode={snapshot.mode.value}; "
            f"detections={detection_text}; "
            f"target={target_text}; "
            f"last_command={snapshot.last_command}; "
            f"fps={snapshot.fps:.1f}; "
            f"inference_ms={snapshot.inference_ms:.1f}; "
            f"servo_pan={snapshot.servo_pan}; "
            f"servo_tilt={snapshot.servo_tilt}; "
            f"connections={', '.join(connection_parts)}."
        )

    @staticmethod
    def _is_scene_question(query: str) -> bool:
        patterns = (
            "what do you see",
            "what can you see",
            "what are you looking at",
            "what's in front",
            "what is in front",
            "describe the scene",
            "scan the scene",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_status_question(query: str) -> bool:
        patterns = (
            "what mode",
            "what are you doing",
            "status",
            "what is your status",
            "are you following",
            "are you autonomous",
            "are you driving yourself",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_connection_question(query: str) -> bool:
        patterns = (
            "camera online",
            "camera working",
            "are you connected",
            "connection status",
            "servo online",
            "motor online",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_target_question(query: str) -> bool:
        patterns = (
            "who are you following",
            "what are you tracking",
            "do you have a target",
            "target locked",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_capability_question(query: str) -> bool:
        patterns = (
            "what can you do",
            "your capabilities",
            "can you help",
            "what do you do",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_help_question(query: str) -> bool:
        patterns = (
            "what commands",
            "how do i command",
            "how can i command",
            "what can i say",
            "voice commands",
            "how do i control you",
            "help me control",
        )
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _is_project_question(query: str) -> bool:
        patterns = (
            "what is this project",
            "what is the project",
            "what is jarvis",
            "who are you",
            "how does this project work",
            "how does the project work",
            "how do you work",
            "how are you built",
            "project architecture",
            "tell me about the project",
            "tell me about jarvis",
        )
        return any(pattern in query for pattern in patterns)

    def _scene_summary(self, snapshot: VisionSnapshot) -> str:
        if snapshot.detections:
            return self._scene_perception.describe(list(snapshot.detections))
        if snapshot.frame is None:
            return "I do not have a live camera frame right now."
        return "I have a live camera feed, but nothing confident is detected in view right now."

    def _command_summary(self) -> str:
        return (
            "You can tell me move forward, move backward, turn left, turn right, stop, "
            "follow me, autonomous mode, manual mode, inspect the scene, or emergency stop."
        )

    def _project_summary(self, query: str) -> str:
        base = (
            "Jarvis is the PC brain for an ESP32-CAM rover. "
            "It pulls the live camera stream into the V.I.S.I.O.N. control app, runs local detection and tracking, "
            "aims the pan tilt servos to keep one target centered, drives the rover motors when commanded, "
            "and adds voice interaction plus the live HUD on top."
        )
        if self._knowledge_base is None:
            return base

        chunks = self._knowledge_base.search(query, limit=4)
        if not chunks:
            return base

        preferred = sorted(
            chunks,
            key=lambda chunk: (
                0 if chunk.path.lower().endswith("readme.md") else 1,
                0 if chunk.path.lower().endswith((".md", ".txt")) else 1,
                len(chunk.text),
            ),
        )
        context_lines: list[str] = []
        for chunk in preferred[:2]:
            sentence = self._normalize_chunk_text(chunk.text)
            if sentence and sentence not in context_lines:
                context_lines.append(sentence)
        if not context_lines:
            return base

        context = " ".join(context_lines)
        merged = f"{base} {context}"
        return merged if len(merged) <= 420 else f"{merged[:417].rsplit(' ', 1)[0]}."

    @classmethod
    def _normalize_chunk_text(cls, text: str) -> str:
        compact = cls._WS_RE.sub(" ", (text or "").strip())
        return compact

    def _status_summary(self, snapshot: VisionSnapshot) -> str:
        mode = snapshot.mode
        if mode == ControlMode.FOLLOW_PERSON:
            if snapshot.target_locked and snapshot.target is not None:
                return "I am in follow mode and I have a person locked in view."
            return "I am in follow mode and waiting for a solid person lock."
        if mode == ControlMode.AUTONOMOUS:
            return "I am in autonomous mode and driving from the live scene feed."
        if mode == ControlMode.INSPECT_SCENE:
            return "I am inspecting the live scene right now."
        if mode == ControlMode.VOICE_NAV:
            return "I am executing a direct voice navigation command."
        if snapshot.last_command and snapshot.last_command != "S":
            return f"I am in manual mode and my last motion command was {snapshot.last_command}."
        return "I am standing by in manual control."

    def _connection_summary(self, snapshot: VisionSnapshot) -> str:
        if not snapshot.links:
            return "I do not have link telemetry yet."
        parts = []
        for channel, state in sorted(snapshot.links.items()):
            state_text = state.value.lower() if isinstance(state, ConnectionState) else str(state).lower()
            parts.append(f"{channel} is {state_text}")
        return ", ".join(parts).capitalize() + "."

    def _target_summary(self, snapshot: VisionSnapshot) -> str:
        if snapshot.target is None:
            return "I do not have a locked target right now."
        label = snapshot.target.label or "target"
        coords = snapshot.target_coords
        if coords:
            return f"I have a {label} locked near {coords[0]}, {coords[1]}."
        return f"I have a {label} locked in view."
