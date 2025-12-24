from __future__ import annotations

from dataclasses import dataclass
from typing import List
import cv2
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
    def __init__(self, cascade_path: str):
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5)
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    @staticmethod
    def annotate_crops(bgr: np.ndarray, crops: List[FaceCrop]) -> np.ndarray:
        out = bgr.copy()
        for c in crops:
            cb = c.crop_box
            cv2.rectangle(out, (cb.x, cb.y), (cb.x + cb.w, cb.y + cb.h), (255, 0, 0), 2)
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