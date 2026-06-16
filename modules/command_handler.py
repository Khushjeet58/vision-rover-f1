from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from core.event_bus import SystemEvents, bus


@dataclass(slots=True)
class CommandResolution:
    command: Optional[str]
    route: str


class CommandHandler:
    VALID_MOVEMENT_COMMANDS = {"F", "B", "L", "R", "S"}
    VALID_META_COMMANDS = {"FOLLOW", "AUTO", "MANUAL", "INSPECT", "E_STOP"}

    SPEECH_MAP = {
        "F": "Moving forward.",
        "B": "Moving backward.",
        "L": "Turning left.",
        "R": "Turning right.",
        "S": "Stopping now.",
        "FOLLOW": "Follow mode is active. I will track the person ahead.",
        "AUTO": "Autonomous mode is active. I will drive from live scene awareness.",
        "MANUAL": "Switching back to manual control.",
        "INSPECT": "Scanning the scene in front of me.",
        "E_STOP": "Emergency stop engaged.",
    }

    FUZZY_TARGETS = {
        "move forward": "F",
        "go ahead": "F",
        "move backward": "B",
        "reverse": "B",
        "turn left": "L",
        "turn right": "R",
        "stop": "S",
        "halt": "S",
        "follow the person ahead of you": "FOLLOW",
        "follow person": "FOLLOW",
        "follow me": "FOLLOW",
        "start following me": "FOLLOW",
        "autonomous mode": "AUTO",
        "drive autonomously": "AUTO",
        "drive yourself": "AUTO",
        "start patrol": "AUTO",
        "manual mode": "MANUAL",
        "go manual": "MANUAL",
        "stop following": "MANUAL",
        "stop autonomous mode": "MANUAL",
        "what's in front of you": "INSPECT",
        "what is in front of you": "INSPECT",
        "inspect the scene": "INSPECT",
        "emergency stop": "E_STOP",
        "stop everything": "E_STOP",
    }

    @staticmethod
    def _log(message: str) -> None:
        bus.emit(SystemEvents.LOG_MESSAGE, f"[COMMAND] {message}")

    @classmethod
    def parse_local_command(cls, user_input: str) -> Optional[str]:
        text = (user_input or "").lower().strip()
        if not text:
            return None

        best_match = None
        best_ratio = 0.0
        for phrase, cmd in cls.FUZZY_TARGETS.items():
            ratio = fuzz.partial_ratio(text, phrase)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cmd

        cls._log(f"Input: '{text}' | Match: {best_match} | Confidence: {best_ratio:.2f}%")
        if best_ratio >= 78:
            return best_match

        if "follow" in text and "person" in text:
            return "FOLLOW"
        if "follow me" in text:
            return "FOLLOW"
        if "autonomous" in text or "autopilot" in text or "drive yourself" in text or "patrol" in text:
            return "AUTO"
        if "manual" in text or "stop following" in text:
            return "MANUAL"
        if "front" in text and ("what" in text or "scan" in text):
            return "INSPECT"
        if "inspect" in text and "scene" in text:
            return "INSPECT"
        if "emergency" in text and "stop" in text:
            return "E_STOP"
        if "stop everything" in text:
            return "E_STOP"
        if "forward" in text or "ahead" in text:
            return "F"
        if "back" in text or "reverse" in text:
            return "B"
        if "left" in text:
            return "L"
        if "right" in text:
            return "R"
        if "stop" in text or "halt" in text:
            return "S"
        return None

    @classmethod
    def speech_for(cls, command: str) -> str:
        return cls.SPEECH_MAP.get(command, "Command executed.")
