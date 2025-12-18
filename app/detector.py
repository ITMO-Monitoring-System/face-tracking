from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    def __init__(self, cascade_path: str):
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5)
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    @staticmethod
    def annotate(bgr: np.ndarray, faces: List[FaceBox]) -> np.ndarray:
        out = bgr.copy()
        for f in faces:
            cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), (255, 0, 0), 2)
        return out

    @staticmethod
    def crop_faces(bgr: np.ndarray, faces: List[FaceBox]) -> List[Tuple[FaceBox, np.ndarray]]:
        crops: List[Tuple[FaceBox, np.ndarray]] = []
        h_img, w_img = bgr.shape[:2]
        for f in faces:
            pad = int(0.1 * min(f.w, f.h))
            x1 = max(0, f.x - pad)
            y1 = max(0, f.y - pad)
            x2 = min(w_img, f.x + f.w + pad)
            y2 = min(h_img, f.y + f.h + pad)
            crop = bgr[y1:y2, x1:x2]
            crops.append((f, crop))
        return crops