from __future__ import annotations

import audioop
import io
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional

import pyaudio

from config import RoverConfig
from core.event_bus import SystemEvents, bus

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


@dataclass(slots=True)
class ClapDetector:
    threshold: float
    window_seconds: float
    cooldown_seconds: float
    min_separation_seconds: float = 0.12
    _first_clap_time: float | None = None
    _cooldown_until: float = 0.0

    def register_peak(self, amplitude: float, now: float) -> bool:
        if now < self._cooldown_until or amplitude < self.threshold:
            return False

        if self._first_clap_time is None or (now - self._first_clap_time) > self.window_seconds:
            self._first_clap_time = now
            return False
        if (now - self._first_clap_time) < self.min_separation_seconds:
            return False

        self._first_clap_time = None
        self._cooldown_until = now + self.cooldown_seconds
        return True


class AudioService:
    """Own the microphone and arbitrate clap wake plus local speech capture."""

    _TTS_COOLDOWN = 1.0

    def __init__(self, config: RoverConfig) -> None:
        self._config = config
        self._listening_enabled = False
        self._wake_enabled = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._launch_callback: Optional[Callable[[], None]] = None
        self._last_tts_done = 0.0
        self._clap_detector = ClapDetector(
            threshold=config.clap_amplitude_threshold,
            window_seconds=config.clap_window_seconds,
            cooldown_seconds=config.clap_cooldown_seconds,
            min_separation_seconds=config.clap_min_separation_seconds,
        )
        self._whisper_model = None
        bus.subscribe(SystemEvents.TTS_FINISHED, self._on_tts_finished)
        bus.subscribe(SystemEvents.MIC_TOGGLE, self.toggle_listening)

    def set_launch_callback(self, callback: Callable[[], None]) -> None:
        self._launch_callback = callback

    def _log(self, message: str) -> None:
        print(message, flush=True)
        bus.emit(SystemEvents.LOG_MESSAGE, message)

    def toggle_listening(self, active: bool) -> None:
        self._listening_enabled = bool(active)
        self._ensure_thread()

    def set_wake_listener(self, active: bool) -> None:
        was_enabled = self._wake_enabled
        self._wake_enabled = bool(active)
        self._ensure_thread()
        if self._wake_enabled != was_enabled:
            status = "enabled" if self._wake_enabled else "disabled"
            self._log(f"[AudioService] Clap wake listener {status}.")

    def stop(self) -> None:
        self._running = False

    def _ensure_thread(self) -> None:
        should_run = self._listening_enabled or self._wake_enabled
        if should_run and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="AudioService_Thread")
            self._thread.start()
        elif not should_run:
            self._running = False

    def _run(self) -> None:
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._config.audio_sample_rate,
                input=True,
                frames_per_buffer=self._config.audio_chunk_size,
            )
        except Exception as exc:
            self._log(f"[AudioService] Microphone unavailable; clap listener cannot start: {exc}")
            self._running = False
            return
        self._log(self._microphone_status_message())

        speech_frames: list[bytes] = []
        speech_active = False
        silence_start = 0.0

        try:
            while self._running:
                if not self._listening_enabled and not self._wake_enabled:
                    time.sleep(0.1)
                    continue

                try:
                    chunk = stream.read(self._config.audio_chunk_size, exception_on_overflow=False)
                except Exception as exc:
                    self._log(f"[AudioService] Stream read error: {exc}")
                    time.sleep(0.2)
                    continue

                now = time.monotonic()
                amplitude = self._normalized_peak(chunk)

                if self._wake_enabled and self._clap_detector.register_peak(amplitude, now):
                    self._log("[AudioService] Double clap detected.")
                    bus.emit(SystemEvents.AUDIO_WAKE_TRIGGERED, "DOUBLE_CLAP")
                    bus.emit(SystemEvents.APP_LAUNCH_REQUESTED, "DOUBLE_CLAP")
                    if self._launch_callback:
                        try:
                            self._launch_callback()
                        except Exception as exc:
                            self._log(f"[AudioService] Launch callback failed: {exc}")

                if not self._listening_enabled or self._in_tts_cooldown():
                    speech_frames.clear()
                    speech_active = False
                    silence_start = 0.0
                    continue

                if amplitude >= self._config.speech_activation_threshold:
                    speech_active = True
                    speech_frames.append(chunk)
                    silence_start = 0.0
                    continue

                if speech_active:
                    speech_frames.append(chunk)
                    if silence_start == 0.0:
                        silence_start = now
                    elif (now - silence_start) >= self._config.speech_silence_seconds:
                        self._emit_transcript(b"".join(speech_frames))
                        speech_frames.clear()
                        speech_active = False
                        silence_start = 0.0
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                audio.terminate()
            except Exception:
                pass

    def _emit_transcript(self, pcm_bytes: bytes) -> None:
        transcript = self._transcribe_pcm(pcm_bytes)
        if transcript:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[User] {transcript}")
            bus.emit(SystemEvents.VOICE_TEXT_CAPTURED, transcript)

    def _transcribe_pcm(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes:
            return ""

        if WhisperModel is None:
            bus.emit(SystemEvents.LOG_MESSAGE, "[AudioService] faster-whisper is not installed.")
            return ""

        if self._whisper_model is None:
            try:
                self._whisper_model = WhisperModel(
                    self._config.stt_model_size,
                    device=self._config.stt_device,
                    compute_type=self._config.stt_compute_type,
                )
            except Exception as exc:
                bus.emit(SystemEvents.LOG_MESSAGE, f"[AudioService] Failed to load whisper model: {exc}")
                return ""

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._config.audio_sample_rate)
            wav_file.writeframes(pcm_bytes)
        wav_io.seek(0)

        try:
            segments, _ = self._whisper_model.transcribe(
                wav_io,
                language=self._config.stt_language,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return transcript
        except Exception as exc:
            bus.emit(SystemEvents.LOG_MESSAGE, f"[AudioService] Transcription failed: {exc}")
            return ""

    def _on_tts_finished(self, _payload=None) -> None:
        self._last_tts_done = time.time()

    def _in_tts_cooldown(self) -> bool:
        return (time.time() - self._last_tts_done) < self._TTS_COOLDOWN

    def _microphone_status_message(self) -> str:
        if self._wake_enabled and self._listening_enabled:
            return "[AudioService] Microphone stream active. Listening for voice commands and double clap."
        if self._wake_enabled:
            return "[AudioService] Microphone stream active. Listening for double clap."
        return "[AudioService] Microphone stream active. Listening for voice commands."

    @staticmethod
    def _normalized_peak(chunk: bytes) -> float:
        peak = audioop.max(chunk, 2) / 32768.0
        return max(0.0, min(1.0, peak))
