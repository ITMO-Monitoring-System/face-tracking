"""IoU-tracker для bbox-лиц.

Цель — сократить публикацию повторяющихся кропов одного и того же лица в RabbitMQ.
На лекции 30 минут со 30 студентами без трекера = десятки тысяч повторных распознаваний
одних и тех же людей. Трекер сопоставляет bbox-ы между кадрами по IoU и публикует
только новые/просроченные треки.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Tuple

from .detector import FaceBox


def _iou(a: FaceBox, b: FaceBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a.w) * max(0, a.h)
    area_b = max(0, b.w) * max(0, b.h)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


@dataclass
class Track:
    id: int
    bbox: FaceBox
    first_seen_ts: float
    last_seen_ts: float
    last_published_ts: float = 0.0
    miss_count: int = 0


@dataclass
class TrackedFace:
    """Результат update(): face + назначенный track_id + флаг должен ли публиковаться."""
    face: FaceBox
    track_id: int
    should_publish: bool


class IoUTracker:
    """Простой жадный IoU-tracker.

    Параметры:
    - iou_threshold: минимальная IoU для матчинга bbox с существующим треком (0..1).
    - republish_interval: сек между повторными публикациями одного трека (после успеха).
    - max_age: сек без видимости до удаления трека из памяти.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        republish_interval: float = 30.0,
        max_age: float = 5.0,
    ) -> None:
        self._iou = float(iou_threshold)
        self._republish = float(republish_interval)
        self._max_age = float(max_age)
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1
        self._lock = Lock()

    def update(self, faces: List[FaceBox], ts: float | None = None) -> List[TrackedFace]:
        """Сопоставить текущие faces с треками. Вернуть TrackedFace со флагом публикации."""
        now = float(ts) if ts is not None else time.time()
        results: List[TrackedFace] = []

        with self._lock:
            # Жадный матчинг: для каждого face берём трек с максимальным IoU выше порога.
            assigned_tracks: set[int] = set()
            face_to_track: List[Tuple[int, int | None, float]] = []  # (face_idx, track_id, iou)

            track_list = list(self._tracks.values())
            for fi, face in enumerate(faces):
                best_tid: int | None = None
                best_iou = self._iou
                for tr in track_list:
                    if tr.id in assigned_tracks:
                        continue
                    iou = _iou(face, tr.bbox)
                    if iou >= best_iou:
                        best_iou = iou
                        best_tid = tr.id
                if best_tid is not None:
                    assigned_tracks.add(best_tid)
                face_to_track.append((fi, best_tid, best_iou))

            for fi, tid, _iou_val in face_to_track:
                face = faces[fi]
                if tid is None:
                    # Новый трек.
                    tid = self._next_id
                    self._next_id += 1
                    tr = Track(
                        id=tid,
                        bbox=face,
                        first_seen_ts=now,
                        last_seen_ts=now,
                    )
                    self._tracks[tid] = tr
                    results.append(TrackedFace(face=face, track_id=tid, should_publish=True))
                else:
                    tr = self._tracks[tid]
                    tr.bbox = face
                    tr.last_seen_ts = now
                    tr.miss_count = 0
                    # Публиковать если ещё ни разу не успешно опубликован,
                    # либо прошло достаточно времени.
                    should = (
                        tr.last_published_ts <= 0.0
                        or (now - tr.last_published_ts) >= self._republish
                    )
                    results.append(TrackedFace(face=face, track_id=tid, should_publish=should))

            # Чистка старых треков.
            to_drop = [tid for tid, tr in self._tracks.items() if (now - tr.last_seen_ts) > self._max_age]
            for tid in to_drop:
                self._tracks.pop(tid, None)

        return results

    def mark_published(self, track_ids: List[int], ts: float | None = None) -> None:
        """Отметить треки как успешно опубликованные — чтобы не публиковать их повторно до republish_interval."""
        if not track_ids:
            return
        now = float(ts) if ts is not None else time.time()
        with self._lock:
            for tid in track_ids:
                tr = self._tracks.get(tid)
                if tr is not None:
                    tr.last_published_ts = now

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def active_tracks(self) -> int:
        with self._lock:
            return len(self._tracks)
