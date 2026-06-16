import threading
import time
from typing import Optional

try:
    import speech_recognition as sr
except ImportError:
    sr = None

from config import Config
from core.event_bus import bus, SystemEvents

class VoiceEngine:
    # Cooldown in seconds to prevent mic re-triggering on Jarvis's own TTS output
    _TTS_COOLDOWN = 1.0

    def __init__(self):
        self._is_listening = False
        self._thread = None
        
        # Speech Recognition
        self._recognizer = sr.Recognizer() if sr else None

        # Timestamp of last TTS completion — used to blank the mic briefly
        self._last_tts_done: float = 0.0

        # Subscribe to know exactly when TTS finishes so we don't transcribe ourselves
        bus.subscribe(SystemEvents.TTS_FINISHED, self._on_tts_finished)

    def _on_tts_finished(self, _payload=None):
        self._last_tts_done = time.time()

    # ------------------------------------------------------------------ #
    #  Toggle API                                                          #
    # ------------------------------------------------------------------ #

    def toggle_listening(self, active: bool):
        """Enable or disable continuous listening."""
        if not sr:
            bus.emit(SystemEvents.LOG_MESSAGE, "[VoiceEngine] speech_recognition missing. Voice disabled.")
            return

        if active and not self._is_listening:
            self._is_listening = True
            self._thread = threading.Thread(target=self._run_continuous_loop, daemon=True)
            self._thread.start()
            
        elif not active and self._is_listening:
            self._is_listening = False
            # Wait for thread to gracefully exit
            if self._thread:
                # We don't join to avoid blocking the main thread, the loop breaks naturally.
                pass

    # ------------------------------------------------------------------ #
    #  Continuous Loop                                                     #
    # ------------------------------------------------------------------ #

    def _run_continuous_loop(self):
        bus.emit(SystemEvents.LOG_MESSAGE, "[VoiceEngine] Microphone is LIVE.")
        
        while self._is_listening:
            if self._in_tts_cooldown():
                time.sleep(0.5)
                continue

            try:
                # Open a fresh mic context for each listening window
                with sr.Microphone() as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    
                    try:
                        # Listen for a phrase. Timeout 1s means if no one speaks for 1s, it loops early.
                        # This ensures the _is_listening flag can break the loop promptly.
                        audio = self._recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        continue # Nobody spoke, that's fine, loop again
                        
                # Ensure we didn't get toggled off while listening
                if not self._is_listening:
                    break

                bus.emit(SystemEvents.STATE_CHANGE, "THINKING")
                bus.emit(SystemEvents.LOG_MESSAGE, "[VoiceEngine] Processing Speech...")
                
                text = self._recognizer.recognize_google(audio)
                
                if text and text.strip():
                    # Check cooldown again just in case TTS suddenly cut in
                    if not self._in_tts_cooldown():
                        bus.emit(SystemEvents.LOG_MESSAGE, f"[User] {text}")
                        bus.emit(SystemEvents.VOICE_TEXT_CAPTURED, text)
                        # We don't need to manually sleep here. The system handles the text, 
                        # triggers TTS (which emits STATE_CHANGE "SPEAKING"), and the while loop
                        # will hit the _in_tts_cooldown() block gracefully!
                else:
                    bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
            except Exception as e:
                bus.emit(SystemEvents.LOG_MESSAGE, f"[VoiceEngine] STT Error: {e}")
                time.sleep(1)
                
        bus.emit(SystemEvents.LOG_MESSAGE, "[VoiceEngine] Microphone is OFF.")
        bus.emit(SystemEvents.STATE_CHANGE, "IDLE")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _in_tts_cooldown(self) -> bool:
        """True if we're too close to the end of the last TTS output to trust mic input."""
        return (time.time() - self._last_tts_done) < self._TTS_COOLDOWN
