from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import FaceBox, FaceDetector


class CameraWorker:
    def __init__(self, camera_id: int, detector: FaceDetector):
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera id={camera_id}")

        self._detector = detector
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._last_frame: Optional[np.ndarray] = None
        self._last_faces: List[FaceBox] = []
        self._last_ts: float = 0.0

        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            faces = self._detector.detect(frame)
            with self._lock:
                self._last_frame = frame
                self._last_faces = faces
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