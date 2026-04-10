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

    Поддерживает тайлинг: кадр разбивается на перекрывающиеся фрагменты,
    каждый обрабатывается отдельно, результаты мержатся через NMS.
    Это позволяет находить мелкие лица в больших кадрах (1080p+).
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_selection: int = 1,
        tile_enabled: bool = False,
        tile_cols: int = 2,
        tile_rows: int = 2,
        tile_overlap: float = 0.3,
        tile_nms_iou: float = 0.4,
        tile_min_face_px: int = 20,
        tile_every_n_frames: int = 1,
    ):
        self._min_confidence = float(min_confidence)
        self._model_selection = int(model_selection)
        self._tile_enabled = tile_enabled
        self._tile_cols = max(1, tile_cols)
        self._tile_rows = max(1, tile_rows)
        self._tile_overlap = max(0.0, min(0.5, tile_overlap))
        self._tile_nms_iou = tile_nms_iou
        self._tile_min_face_px = tile_min_face_px
        self._tile_every_n_frames = max(1, tile_every_n_frames)
        self._frame_counter = 0

        _mp_face = mp.solutions.face_detection
        self._detector = _mp_face.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_confidence,
        )

    def _detect_single(self, bgr: np.ndarray) -> List[FaceBox]:
        """Run MediaPipe on a single image region."""
        h_img, w_img = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces: List[FaceBox] = []
        if not results.detections:
            return faces

        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = int(max(0.0, bb.xmin) * w_img)
            y = int(max(0.0, bb.ymin) * h_img)
            fw = int(min(bb.width, 1.0 - max(0.0, bb.xmin)) * w_img)
            fh = int(min(bb.height, 1.0 - max(0.0, bb.ymin)) * h_img)
            if fw > 0 and fh > 0:
                faces.append(FaceBox(x, y, fw, fh))

        return faces

    @staticmethod
    def _iou(a: FaceBox, b: FaceBox) -> float:
        """Intersection over Union for two bounding boxes."""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = a.w * a.h
        area_b = b.w * b.h
        return inter / (area_a + area_b - inter)

    def _nms(self, faces: List[FaceBox]) -> List[FaceBox]:
        """Non-maximum suppression: remove duplicate detections from overlapping tiles."""
        if len(faces) <= 1:
            return faces
        # Sort by area descending (prefer larger detections)
        sorted_faces = sorted(faces, key=lambda f: f.w * f.h, reverse=True)
        keep: List[FaceBox] = []
        for face in sorted_faces:
            if all(self._iou(face, kept) < self._tile_nms_iou for kept in keep):
                keep.append(face)
        return keep

    def _generate_tiles(self, h: int, w: int) -> List[tuple]:
        """Generate (x_offset, y_offset, tile_w, tile_h) for overlapping tiles."""
        cols = self._tile_cols
        rows = self._tile_rows
        overlap = self._tile_overlap

        tile_w = int(w / (cols - overlap * (cols - 1))) if cols > 1 else w
        tile_h = int(h / (rows - overlap * (rows - 1))) if rows > 1 else h
        # Ensure tiles aren't larger than the frame
        tile_w = min(tile_w, w)
        tile_h = min(tile_h, h)

        step_x = int(tile_w * (1.0 - overlap)) if cols > 1 else 0
        step_y = int(tile_h * (1.0 - overlap)) if rows > 1 else 0

        tiles = []
        for r in range(rows):
            for c in range(cols):
                x_off = min(c * step_x, w - tile_w) if step_x > 0 else 0
                y_off = min(r * step_y, h - tile_h) if step_y > 0 else 0
                tiles.append((x_off, y_off, tile_w, tile_h))
        return tiles

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        # Always run full-frame detection
        full_faces = self._detect_single(bgr)

        if not self._tile_enabled:
            return full_faces

        # Tiled detection only every N frames to reduce CPU load
        self._frame_counter += 1
        if self._frame_counter % self._tile_every_n_frames != 0:
            return full_faces

        # Tiled detection for small faces
        h_img, w_img = bgr.shape[:2]
        tiles = self._generate_tiles(h_img, w_img)
        tile_faces: List[FaceBox] = []

        for x_off, y_off, tw, th in tiles:
            tile = bgr[y_off:y_off + th, x_off:x_off + tw]
            detected = self._detect_single(tile)
            for f in detected:
                # Translate to full-frame coordinates
                global_face = FaceBox(
                    x=f.x + x_off, y=f.y + y_off, w=f.w, h=f.h,
                )
                # Skip very small detections (likely noise)
                if global_face.w >= self._tile_min_face_px and global_face.h >= self._tile_min_face_px:
                    tile_faces.append(global_face)

        # Merge full-frame and tiled detections via NMS
        all_faces = full_faces + tile_faces
        return self._nms(all_faces)

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
