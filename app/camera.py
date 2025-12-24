from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .detector import FaceBox, FaceDetector


def _parse_source(src: str) -> Union[int, str]:
    s = (src or "").strip()
    if s.lower() in ("none", "off", "disabled", ""):
        return "none"
    # "0", "1", "2" -> int camera id
    if s.isdigit():
        return int(s)
    return s  # rtsp/http file path etc


class CameraWorker:
    """
    Camera worker that:
    - supports int camera id OR rtsp url via CAMERA_SOURCE
    - doesn't crash if source is unavailable
    - reconnects in background
    """

    def __init__(self, source: str, detector: FaceDetector, reconnect_interval: float = 1.0):
        self._source_raw = source
        self._source = _parse_source(source)
        self._reconnect_interval = float(reconnect_interval)

        self._cap: Optional[cv2.VideoCapture] = None
        self._detector = detector

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._last_frame: Optional[np.ndarray] = None
        self._last_faces: List[FaceBox] = []
        self._last_ts: float = 0.0

        self._thread = threading.Thread(target=self._loop, daemon=True)

    @property
    def source(self) -> str:
        return str(self._source_raw)

    def enabled(self) -> bool:
        return self._source != "none"

    def start(self) -> None:
        if not self.enabled():
            return
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        if self._source == "none":
            return None

        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def _loop(self) -> None:
        fail_count = 0

        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                print(f"Opening capture: {self._source}")
                self._cap = self._open_capture()
                if self._cap is None:
                    print(f"Failed to open capture, retrying in {self._reconnect_interval}s")
                    time.sleep(self._reconnect_interval)
                    continue
                fail_count = 0

            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail_count += 1
                print(f"Failed to read frame ({fail_count})")
                time.sleep(0.05)
                if fail_count >= 300:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                continue

            fail_count = 0
            print(f"Frame captured: {frame.shape if frame is not None else 'None'}")

            with self._lock:
                self._last_frame = frame
                self._last_faces = []
                self._last_ts = time.time()

    def snapshot(self) -> Tuple[Optional[np.ndarray], List[FaceBox], float]:
        with self._lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
            faces = list(self._last_faces)
            ts = float(self._last_ts)
        return frame, faces, ts

    @staticmethod
    def encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("Failed to encode jpeg")
        return buf.tobytes()
