import queue
import time
import threading
from pathlib import Path

import numpy as np
import pyaudio
from runtime_preload import preload_onnxruntime

# Piper model path relative to project root
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_ONNX_MODEL = _MODEL_DIR / "en_US-lessac-medium.onnx"

_PIPER_AVAILABLE = _ONNX_MODEL.exists()

class TTSEngine:
    """
    Non-blocking TTS engine.
    Primary:  Piper TTS (neural, natural-sounding, fully offline)
    Fallback: pyttsx3 (SAPI5 / eSpeak)
    """
    _instance = None
    _lock = threading.Lock()
    _queue: queue.Queue = queue.Queue()
    _queued_texts: set[str] = set()
    _is_speaking: bool = False
    _cancel_current: bool = False
    _current_text: str = ""
    _thread: threading.Thread = None

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._thread = threading.Thread(target=cls._process_queue_loop, daemon=True)
                cls._thread.start()
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def speak(cls, text: str, on_start=None, on_done=None, interrupt=False):
        """Queue a string to speak asynchronously. If interrupt is True, stops current speech first."""
        if not text:
            cls._safe_call(on_start)
            cls._safe_call(on_done)
            return

        line = " ".join(str(text).split())
        if not line:
            cls._safe_call(on_start)
            cls._safe_call(on_done)
            return

        with cls._lock:
            if (cls._is_speaking and cls._current_text == line) or line in cls._queued_texts:
                cls._safe_call(on_done)
                return

        if interrupt:
            cls.interrupt()

        with cls._lock:
            cls._queued_texts.add(line)
        cls._queue.put((line, on_start, on_done))

    @classmethod
    def is_speaking(cls) -> bool:
        return cls._is_speaking

    @classmethod
    def has_pending(cls) -> bool:
        with cls._lock:
            return cls._is_speaking or bool(cls._queued_texts)

    @classmethod
    def interrupt(cls):
        """Clear the queue and signal the active synthesizer to abort."""
        with cls._lock:
            if cls._is_speaking:
                cls._cancel_current = True
            cls._queued_texts.clear()
            
        while not cls._queue.empty():
            try:
                text, st, dn = cls._queue.get_nowait()
                cls._safe_call(dn)
                cls._queue.task_done()
            except queue.Empty:
                break

    # ------------------------------------------------------------------ #
    #  Worker loop                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def _process_queue_loop(cls):
        print("[TTS/Thread] Worker thread started!")
        # CoInitialize required for SAPI5 on Windows background threads
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        if _PIPER_AVAILABLE:
            print("[TTS/Thread] Routing to Piper loop...")
            try:
                cls._run_piper_loop()
            except Exception as e:
                print(f"[TTS/Thread] Piper loop crashed: {e}")
                print("[TTS/Thread] Falling back to pyttsx3 loop...")
                cls._run_pyttsx3_loop()
        else:
            print("[TTS/Thread] Routing to pyttsx3 loop...")
            cls._run_pyttsx3_loop()

    @classmethod
    def _run_piper_loop(cls):
        print("[TTS/Piper] Importing PiperVoice...")
        if not preload_onnxruntime():
            raise ImportError("onnxruntime preload failed")

        import onnxruntime  # noqa: F401
        from piper import PiperVoice

        print(f"[TTS/Piper] Loading model: {_ONNX_MODEL}")
        voice = PiperVoice.load(str(_ONNX_MODEL))
        print("[TTS/Piper] Initializing PyAudio...")
        pa = pyaudio.PyAudio()

        print("[TTS/Piper] Ready. Waiting for queue...")
        while True:
            text, on_start, on_done = cls._queue.get()

            with cls._lock:
                cls._queued_texts.discard(text)
                cls._is_speaking = True
                cls._current_text = text
                cls._cancel_current = False
                
            cls._safe_call(on_start)

            try:
                print(f"[TTS/Piper] Generating audio for: {text}")
                stream = None
                for chunk in voice.synthesize(text):
                    if cls._cancel_current:
                        break
                    samples = np.asarray(chunk.audio_float_array, dtype=np.float32)
                    if not samples.size:
                        continue
                    if stream is None:
                        stream = pa.open(
                            format=pyaudio.paFloat32,
                            channels=1,
                            rate=voice.config.sample_rate,
                            output=True,
                        )
                    if cls._cancel_current:
                        break
                    stream.write(samples.tobytes())
                if stream is not None:
                    try:
                        stream.stop_stream()
                    finally:
                        stream.close()
                if not cls._cancel_current:
                    print(f"[TTS/Piper] Finished playing: {text}")
            except Exception as e:
                print(f"[TTS/Piper] Error: {e}")

            with cls._lock:
                cls._cancel_current = False
                cls._is_speaking = False
                cls._current_text = ""
                
            cls._safe_call(on_done)
            time.sleep(0.2)   # Brief cooldown so mic doesn't capture TTS tail
            cls._queue.task_done()

    @classmethod
    def _run_pyttsx3_loop(cls):
        """Fallback path: pyttsx3 (SAPI5 on Windows)."""
        import pyttsx3

        while True:
            text, on_start, on_done = cls._queue.get()

            with cls._lock:
                cls._queued_texts.discard(text)
                cls._is_speaking = True
                cls._current_text = text
                cls._cancel_current = False
                
            cls._safe_call(on_start)

            try:
                if not cls._cancel_current:
                    # Initialize pyttsx3 FRESH for each utterance to bypass SAPI5 thread freezes
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 170)
                    engine.setProperty('volume', 1.0)
                    voices = engine.getProperty('voices')
                    chosen = voices[0].id if voices else None
                    for v in voices:
                        if any(n in v.name for n in ('David', 'Mark', 'George', 'Male')):
                            chosen = v.id
                            break
                    if chosen:
                        engine.setProperty('voice', chosen)
                    
                    engine.say(text)
                    engine.runAndWait()
                    del engine # Force cleanup of COM objects
            except Exception as e:
                print(f"[TTS/pyttsx3] Speak failed: {e}")

            with cls._lock:
                cls._cancel_current = False
                cls._is_speaking = False
                cls._current_text = ""
                
            cls._safe_call(on_done)
            time.sleep(0.2)   # Brief cooldown so mic doesn't capture TTS tail
            cls._queue.task_done()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_call(fn):
        if fn:
            try:
                fn()
            except Exception:
                pass
