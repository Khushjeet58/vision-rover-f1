from config import rover_config
from core.event_bus import SystemEvents, bus
from modules.audio_service import AudioService, ClapDetector


def test_clap_detector_requires_two_peaks_within_window():
    detector = ClapDetector(threshold=0.5, window_seconds=0.8, cooldown_seconds=1.0)

    assert detector.register_peak(0.6, 1.0) is False
    assert detector.register_peak(0.4, 1.2) is False
    assert detector.register_peak(0.7, 1.4) is True


def test_clap_detector_respects_cooldown():
    detector = ClapDetector(threshold=0.5, window_seconds=0.8, cooldown_seconds=1.0)
    detector.register_peak(0.7, 1.0)
    assert detector.register_peak(0.7, 1.4) is True
    assert detector.register_peak(0.8, 1.6) is False


def test_clap_detector_ignores_single_clap_echo_peak():
    detector = ClapDetector(
        threshold=0.5,
        window_seconds=0.8,
        cooldown_seconds=1.0,
        min_separation_seconds=0.16,
    )

    assert detector.register_peak(0.7, 1.0) is False
    assert detector.register_peak(0.7, 1.05) is False
    assert detector.register_peak(0.7, 1.30) is True


def test_wake_listener_enables_audio_thread_gate(monkeypatch):
    service = AudioService(rover_config)
    ensure_calls = []

    def fake_ensure_thread():
        ensure_calls.append((service._wake_enabled, service._listening_enabled))

    monkeypatch.setattr(service, "_ensure_thread", fake_ensure_thread)

    service.set_wake_listener(True)
    service.set_wake_listener(False)

    assert ensure_calls == [(True, False), (False, False)]


def test_microphone_status_message_does_not_claim_double_clap_when_wake_disabled():
    service = AudioService(rover_config)

    service._listening_enabled = True
    service._wake_enabled = False

    assert service._microphone_status_message() == "[AudioService] Microphone stream active. Listening for voice commands."


def test_audio_service_only_starts_tts_cooldown_after_real_speaking_cycle():
    service = AudioService(rover_config)
    original = service._last_tts_done

    bus.emit(SystemEvents.STATE_CHANGE, "IDLE")
    assert service._last_tts_done == original

    bus.emit(SystemEvents.TTS_FINISHED, "hello")

    assert service._last_tts_done >= original
