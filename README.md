# Vision Rover Final

Vision Rover Final is the PC-side autonomy stack for a student-built ESP32 rover. It combines live ESP32-CAM video, YOLO-based human tracking, pan/tilt servo control, rover drive control, voice interaction, and a custom V.I.S.I.O.N operator HUD.

## Highlights

- Real-time camera streaming with latest-frame-first handling for low-latency tracking.
- Face-prioritized person lock and follow behavior.
- Pan/tilt servo aiming with smoothing, deadband, Kalman prediction, and PID control.
- Manual, follow, and autonomous operating modes.
- Local voice commands, local TTS, and local Ollama-backed project assistant responses.
- Modular Python architecture with tests.

## Tech Stack

- Python 3.11
- PyQt5
- OpenCV
- Ultralytics YOLO
- PyTorch with CUDA support
- Faster-Whisper
- Piper / offline TTS
- Ollama for local LLM responses

## Run On Windows

```powershell
git clone https://github.com/Khushjeet58/vision-rover-f1.git
cd vision-rover-final
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe launcher.py
```

## Local Configuration

This public repo keeps machine-specific IPs and runtime settings out of tracked source.

1. Create `local_config/.env`.
2. Start from `local_config_example/.env.example`.
3. Replace the placeholder rover and camera values with your own.

You can also override with shell environment variables:

```powershell
$env:ROVER_CAMERA_IP="192.168.0.100"
$env:ROVER_ESP32_IP="192.168.0.101"
$env:VISION_PERF_PROFILE="rtx5060"
$env:OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
```

## Notes

- Local virtual environments, caches, and local config are intentionally excluded from Git.
- YOLO `.pt` weights are intentionally ignored and can be downloaded locally as needed.
- The offline Piper voice model under `models/` remains in the project so the demo works immediately after clone.
