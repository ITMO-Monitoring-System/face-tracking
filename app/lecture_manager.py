import threading
from typing import Optional

from .camera import CameraWorker
from .detector import FaceDetector


class LectureManager:
    def __init__(self, detector: FaceDetector, reconnect_interval: float):
        self._lock = threading.Lock()
        self._cams: dict[str, CameraWorker] = {}
        self._detector = detector
        self._reconnect_interval = reconnect_interval

    def start_camera(self, lecture_id: str, source: str) -> CameraWorker:
        with self._lock:
            cam = self._cams.get(lecture_id)
            if cam:
                return cam

            cam = CameraWorker(
                source=source,
                detector=self._detector,
                reconnect_interval=self._reconnect_interval,
            )
            cam.start()
            self._cams[lecture_id] = cam
            return cam

    def stop_camera(self, lecture_id: str) -> None:
        with self._lock:
            cam = self._cams.pop(lecture_id, None)
        if cam:
            cam.stop()

    def stop_all(self) -> None:
        with self._lock:
            items = list(self._cams.items())
            self._cams.clear()
        for _, cam in items:
            cam.stop()

    def get(self, lecture_id: str) -> Optional[CameraWorker]:
        with self._lock:
            return self._cams.get(lecture_id)
