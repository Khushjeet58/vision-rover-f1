from __future__ import annotations

import argparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


def iter_latest_jpegs(url: str, *, read_size: int = 8192, timeout: float = 5.0):
    headers = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "User-Agent": "VISION-LatestFrameViewer/1.0",
    }
    req = Request(url, headers=headers)
    buffer = bytearray()

    with urlopen(req, timeout=timeout) as stream:
        while True:
            chunk = stream.read(read_size)
            if not chunk:
                raise ConnectionError("MJPEG stream ended")

            buffer.extend(chunk)
            latest_payload: bytes | None = None
            while True:
                soi = buffer.find(b"\xff\xd8")
                if soi < 0:
                    if len(buffer) > 256_000:
                        del buffer[:-2]
                    break
                if soi > 0:
                    del buffer[:soi]

                eoi = buffer.find(b"\xff\xd9", 2)
                if eoi < 0:
                    if len(buffer) > 1_000_000:
                        del buffer[: len(buffer) - 256_000]
                    break

                latest_payload = bytes(buffer[: eoi + 2])
                del buffer[: eoi + 2]

            if latest_payload is not None:
                yield latest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-latency ESP32-CAM MJPEG viewer.")
    parser.add_argument("--url", default="http://192.168.0.100:81/stream")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    scale = max(1.0, float(args.scale))
    for payload in iter_latest_jpegs(args.url):
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        cv2.imshow("ESP32-CAM Latest Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
