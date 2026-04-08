from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class FaceCrop:
    # face: bbox лица в координатах исходного кадра
    face: FaceBox
    # crop_box: bbox вырезки (голова/контекст) в координатах исходного кадра
    crop_box: FaceBox
    # face_in_crop: bbox лица в координатах уже отправляемого crop-изображения
    face_in_crop: FaceBox
    # image: сам crop (BGR)
    image: np.ndarray


class FaceDetector:
    """
    Детектор лиц на основе MediaPipe Face Detection.

    model_selection=1 — full-range модель (до 5 м), оптимальна для аудитории.
    model_selection=0 — short-range (до 2 м), чуть быстрее, для съёмки вблизи.
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        self._min_confidence = float(min_confidence)
        self._model_selection = int(model_selection)

        _mp_face = mp.solutions.face_detection
        self._detector = _mp_face.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_confidence,
        )

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        h_img, w_img = bgr.shape[:2]

        # MediaPipe требует RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces: List[FaceBox] = []
        if not results.detections:
            return faces

        for det in results.detections:
            bb = det.location_data.relative_bounding_box

            # Приводим относительные координаты к пикселям, клипуем по границам
            x = int(max(0.0, bb.xmin) * w_img)
            y = int(max(0.0, bb.ymin) * h_img)
            fw = int(min(bb.width, 1.0 - max(0.0, bb.xmin)) * w_img)
            fh = int(min(bb.height, 1.0 - max(0.0, bb.ymin)) * h_img)

            if fw > 0 and fh > 0:
                faces.append(FaceBox(x, y, fw, fh))

        return faces

    @staticmethod
    def annotate(bgr: np.ndarray, faces: List[FaceBox]) -> np.ndarray:
        out = bgr.copy()
        for f in faces:
            cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 220, 0), 2)
        return out

    @staticmethod
    def crop_faces(
            bgr: np.ndarray,
            faces: List[FaceBox],
            *,
            pad_x_ratio: float = 0.35,
            pad_y_ratio: float = 0.35,
            top_extra_ratio: float = 0.60,
            bottom_extra_ratio: float = 0.25,
            make_square: bool = True,
    ) -> List[FaceCrop]:
        crops: List[FaceCrop] = []
        h_img, w_img = bgr.shape[:2]

        for f in faces:
            pad_x = int(max(0.0, float(pad_x_ratio)) * f.w)
            pad_y = int(max(0.0, float(pad_y_ratio)) * f.h)
            top_extra = int(max(0.0, float(top_extra_ratio)) * f.h)
            bottom_extra = int(max(0.0, float(bottom_extra_ratio)) * f.h)

            x1 = max(0, f.x - pad_x)
            x2 = min(w_img, f.x + f.w + pad_x)
            y1 = max(0, f.y - pad_y - top_extra)
            y2 = min(h_img, f.y + f.h + pad_y + bottom_extra)

            if make_square:
                cw = x2 - x1
                ch = y2 - y1
                if cw > 0 and ch > 0:
                    if cw > ch:
                        need = cw - ch
                        up = need // 2
                        down = need - up
                        y1 = max(0, y1 - up)
                        y2 = min(h_img, y2 + down)
                    elif ch > cw:
                        need = ch - cw
                        left = need // 2
                        right = need - left
                        x1 = max(0, x1 - left)
                        x2 = min(w_img, x2 + right)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_img = bgr[y1:y2, x1:x2]
            crop_box = FaceBox(x=x1, y=y1, w=(x2 - x1), h=(y2 - y1))
            face_in_crop = FaceBox(x=f.x - x1, y=f.y - y1, w=f.w, h=f.h)

            crops.append(FaceCrop(face=f, crop_box=crop_box, face_in_crop=face_in_crop, image=crop_img))

        return crops
