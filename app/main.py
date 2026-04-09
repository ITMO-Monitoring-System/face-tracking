from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .camera import CameraWorker
from .config import settings
from .detector import FaceDetector
from .rabbitmq import FacePublisher

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://projctviscon.vps.webdock.cloud",
        "http://projctviscon.vps.webdock.cloud",
        "http://localhost",
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

detector = FaceDetector(
    min_confidence=settings.face_min_confidence,
    model_selection=settings.face_model_selection,
)

camera = CameraWorker(
    settings.camera_source,
    detector,
    reconnect_interval=settings.camera_reconnect_interval,
)

publisher = FacePublisher(
    settings.rabbitmq_url,
    settings.exchange_name,
    lecture_queue_template=settings.lecture_queue_template,
    lecture_routing_key_template=settings.lecture_routing_key_template,
)


# ──────────────────────────────────────────────
# Pydantic-модели запросов
# ──────────────────────────────────────────────

class LectureStartRequest(BaseModel):
    durable: bool = True
    auto_delete: bool = False
    expires_ms: int | None = None
    message_ttl_ms: int | None = None


class LectureEndRequest(BaseModel):
    if_unused: bool = False
    if_empty: bool = False


class CameraSourceRequest(BaseModel):
    source: str  # "rtsp://...", "0", "1", "none"


def _normalize_source_value(source: Any) -> str:
    if source is None:
        return ""
    return str(source).strip()


# ──────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    camera.start()
    await publisher.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    camera.stop()
    await publisher.close()


# ──────────────────────────────────────────────
# Базовые эндпоинты
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health() -> dict:
    frame, _, ts = camera.snapshot()
    return {
        "ok": True,
        "camera_source": camera.source,
        "camera_enabled": camera.enabled(),
        "camera_has_frame": frame is not None,
        "camera_frame_ts": ts if frame is not None else None,
    }


# ──────────────────────────────────────────────
# Camera API
# ──────────────────────────────────────────────

@app.get("/api/cameras")
async def list_cameras() -> dict:
    """
    Возвращает текущий источник и список доступных камер.
    Список берётся из CAMERA_SOURCES (JSON-массив строк в env).
    """
    frame, _, _ = camera.snapshot()
    current = _normalize_source_value(camera.source)
    sources = [
        _normalize_source_value(source)
        for source in settings.camera_sources
        if _normalize_source_value(source)
    ]
    if current and current not in sources:
        sources.insert(0, current)
    return {
        "current": current,
        "enabled": camera.enabled(),
        "has_frame": frame is not None,
        "sources": sources,
    }


@app.get("/api/camera/current")
async def get_current_camera() -> dict:
    """Текущий источник камеры и её статус."""
    frame, _, ts = camera.snapshot()
    return {
        "source": _normalize_source_value(camera.source),
        "enabled": camera.enabled(),
        "has_frame": frame is not None,
        "frame_ts": ts if frame is not None else None,
    }


@app.put("/api/camera/source")
async def set_camera_source(req: CameraSourceRequest) -> dict:
    """
    Переключает камеру на новый источник в рантайме.
    Принимает: { "source": "rtsp://...", или "0", "1", "none" }
    """
    requested_source = _normalize_source_value(req.source)
    old_source = _normalize_source_value(camera.source)

    # Запускаем switch_source в thread-pool, т.к. join() блокирует event loop
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, camera.switch_source, requested_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch camera: {e}")

    return {
        "ok": True,
        "old_source": old_source,
        "new_source": _normalize_source_value(camera.source),
        "enabled": camera.enabled(),
    }


# ──────────────────────────────────────────────
# Lectures API
# ──────────────────────────────────────────────

@app.get("/api/lectures")
async def list_lectures() -> dict:
    return {"active": publisher.active_lectures()}


async def _wait_for_camera_frame(*, timeout_seconds: float) -> tuple[Any, float]:
    deadline = asyncio.get_event_loop().time() + max(0.0, float(timeout_seconds))
    while True:
        frame, _, ts = camera.snapshot()
        if frame is not None:
            return frame, ts

        if asyncio.get_event_loop().time() >= deadline:
            raise RuntimeError("camera_not_ready")

        await asyncio.sleep(0.05)


@app.post("/api/lectures/{lecture_id}/start")
async def start_lecture(lecture_id: str, req: LectureStartRequest | None = None) -> dict:
    req = req or LectureStartRequest()
    binding = await publisher.start_lecture(
        lecture_id,
        durable=req.durable,
        auto_delete=req.auto_delete,
        expires_ms=req.expires_ms,
        message_ttl_ms=req.message_ttl_ms,
    )

    probe_sent = False
    camera_warning: str | None = None
    try:
        frame, ts = await _wait_for_camera_frame(timeout_seconds=settings.lecture_start_ready_timeout_seconds)
        jpeg = camera.encode_jpeg(frame, settings.jpeg_quality)
        await publisher.publish_face_jpeg(
            lecture_id,
            jpeg,
            metadata={
                "type": "probe",
                "ts": ts,
                "camera_source": camera.source,
                "lecture_id": lecture_id,
            },
        )
        probe_sent = True
    except Exception as e:
        camera_warning = f"camera_not_ready: {e}"

    try:
        async with httpx.AsyncClient(timeout=settings.connect_service_timeout_seconds) as client:
            in_amqp_url = settings.connect_service_in_amqp_url
            parsed = urlparse(in_amqp_url)
            in_host = parsed.hostname

            if in_host in {"rabbitmq_face_tracking", "localhost", "127.0.0.1"}:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "connect_service_in_amqp_url_not_reachable: "
                        f"in_amqp_url={in_amqp_url}. "
                        "Set CONNECT_SERVICE_IN_AMQP_URL to a public IP/domain and exposed port "
                        "(e.g. amqp://guest:guest@projctviscon.vps.webdock.cloud:5673/)."
                    ),
                )

            try:
                lecture_id_for_connect = int(lecture_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="lecture_id_must_be_integer")

            payload = {
                "lecture_id": lecture_id_for_connect,
                "in_amqp_url": in_amqp_url,
                "in_queue": binding.queue_name,
                "threshold": settings.connect_service_threshold,
            }

            resp = await client.post(settings.connect_service_url, json=payload)

            if resp.is_error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"connect_service_error: endpoint={settings.connect_service_url}, "
                        f"in_amqp_url={in_amqp_url}, in_queue={binding.queue_name}; "
                        f"HTTP {resp.status_code}: {resp.text}"
                    ),
                )
    except HTTPException:
        await publisher.end_lecture(lecture_id)
        raise
    except Exception as e:
        await publisher.end_lecture(lecture_id)
        raise HTTPException(
            status_code=502,
            detail=(
                f"connect_service_error: endpoint={settings.connect_service_url}, "
                f"in_amqp_url={settings.connect_service_in_amqp_url}, "
                f"in_queue={binding.queue_name}; {e}"
            ),
        )

    return {
        "ok": True,
        "lecture_id": lecture_id,
        "queue": binding.queue_name,
        "routing_key": binding.routing_key,
        "notified_external_service": True,
        "probe_sent": probe_sent,
        "camera_ready": probe_sent,
        "camera_source": _normalize_source_value(camera.source),
        "warning": camera_warning,
    }


@app.post("/api/lectures/{lecture_id}/end")
async def end_lecture(lecture_id: str, req: LectureEndRequest | None = None) -> dict:
    req = req or LectureEndRequest()
    deleted = await publisher.end_lecture(lecture_id, if_unused=req.if_unused, if_empty=req.if_empty)
    return {"ok": True, "lecture_id": lecture_id, "deleted": deleted}


# ──────────────────────────────────────────────
# Face publishing logic
# ──────────────────────────────────────────────

async def publish_current_faces(lecture_id: str) -> dict:
    frame, _, ts = camera.snapshot()
    if frame is None:
        return {"published": 0, "error": "no_frame_yet", "lecture_id": lecture_id}

    faces = detector.detect(frame)

    if len(faces) == 0:
        return {"published": 0, "faces": 0, "ts": ts, "lecture_id": lecture_id}

    # TEST MODE: публикуем полный кадр
    if getattr(settings, "publish_mode", "faces") == "frame":
        h, w = frame.shape[:2]
        jpeg = camera.encode_jpeg(frame, settings.jpeg_quality)

        metadata = {
            "type": "full_frame",
            "ts": ts,
            "camera_source": camera.source,
            "lecture_id": lecture_id,
            "idx": 0,
            "bbox": [0, 0, w, h],
            "faces_bboxes": [[f.x, f.y, f.w, f.h] for f in faces],
            "faces": len(faces),
            "frame_wh": [w, h],
        }

        await publisher.publish_face_jpeg(lecture_id, jpeg, metadata=metadata)
        return {"published": 1, "faces": len(faces), "ts": ts, "lecture_id": lecture_id, "mode": "frame"}

    # DEFAULT MODE: кропы лиц
    crops = detector.crop_faces(
        frame,
        faces,
        pad_x_ratio=settings.face_pad_x_ratio,
        pad_y_ratio=settings.face_pad_y_ratio,
        top_extra_ratio=settings.face_top_extra_ratio,
        bottom_extra_ratio=settings.face_bottom_extra_ratio,
        make_square=settings.face_crop_make_square,
    )

    h_frame, w_frame = frame.shape[:2]
    published = 0

    for idx, fc in enumerate(crops):
        crop = fc.image
        h_crop, w_crop = crop.shape[:2]

        jpeg = camera.encode_jpeg(crop, settings.jpeg_quality)

        metadata = {
            "type": "face_head_crop",
            "ts": ts,
            "camera_source": camera.source,
            "idx": idx,
            "lecture_id": lecture_id,
            "bbox": [0, 0, w_crop, h_crop],
            "face_bbox": [fc.face_in_crop.x, fc.face_in_crop.y, fc.face_in_crop.w, fc.face_in_crop.h],
            "original_bbox": [fc.face.x, fc.face.y, fc.face.w, fc.face.h],
            "crop_bbox": [fc.crop_box.x, fc.crop_box.y, fc.crop_box.w, fc.crop_box.h],
            "frame_wh": [w_frame, h_frame],
            "crop_wh": [w_crop, h_crop],
        }

        await publisher.publish_face_jpeg(lecture_id, jpeg, metadata=metadata)
        published += 1

    return {"published": published, "faces": len(faces), "ts": ts, "lecture_id": lecture_id, "mode": "faces"}


# ──────────────────────────────────────────────
# WebSocket stream
# ──────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    import cv2
    import numpy as np

    lecture_id = ws.query_params.get("lecture_id")
    await ws.accept()

    if not lecture_id:
        await ws.send_text(json.dumps({"type": "error", "error": "lecture_id_required"}))
        await ws.close(code=1008)
        return

    if not publisher.is_lecture_active(lecture_id):
        await ws.send_text(json.dumps({"type": "error", "error": "lecture_not_started", "lecture_id": lecture_id}))
        await ws.close(code=1008)
        return

    # Shared state for browser-camera mode
    browser_frame = None  # latest decoded frame from browser
    browser_frame_ts: float = 0.0
    frame_lock = asyncio.Lock()

    last_auto_publish = 0.0
    last_face_count = 0

    async def _publish_faces_from_frame(frame, ts: float) -> dict:
        """Detect faces in frame and publish crops to RabbitMQ."""
        faces = detector.detect(frame)
        if len(faces) == 0:
            return {"published": 0, "faces": 0, "ts": ts, "lecture_id": lecture_id}

        if getattr(settings, "publish_mode", "faces") == "frame":
            h, w = frame.shape[:2]
            jpeg = camera.encode_jpeg(frame, settings.jpeg_quality)
            metadata = {
                "type": "full_frame",
                "ts": ts,
                "camera_source": "browser",
                "lecture_id": lecture_id,
                "idx": 0,
                "bbox": [0, 0, w, h],
                "faces_bboxes": [[f.x, f.y, f.w, f.h] for f in faces],
                "faces": len(faces),
                "frame_wh": [w, h],
            }
            await publisher.publish_face_jpeg(lecture_id, jpeg, metadata=metadata)
            return {"published": 1, "faces": len(faces), "ts": ts, "lecture_id": lecture_id, "mode": "frame"}

        crops = detector.crop_faces(
            frame, faces,
            pad_x_ratio=settings.face_pad_x_ratio,
            pad_y_ratio=settings.face_pad_y_ratio,
            top_extra_ratio=settings.face_top_extra_ratio,
            bottom_extra_ratio=settings.face_bottom_extra_ratio,
            make_square=settings.face_crop_make_square,
        )
        h_frame, w_frame = frame.shape[:2]
        published = 0
        for idx, fc in enumerate(crops):
            crop = fc.image
            h_crop, w_crop = crop.shape[:2]
            jpeg = camera.encode_jpeg(crop, settings.jpeg_quality)
            metadata = {
                "type": "face_head_crop",
                "ts": ts,
                "camera_source": "browser",
                "idx": idx,
                "lecture_id": lecture_id,
                "bbox": [0, 0, w_crop, h_crop],
                "face_bbox": [fc.face_in_crop.x, fc.face_in_crop.y, fc.face_in_crop.w, fc.face_in_crop.h],
                "original_bbox": [fc.face.x, fc.face.y, fc.face.w, fc.face.h],
                "crop_bbox": [fc.crop_box.x, fc.crop_box.y, fc.crop_box.w, fc.crop_box.h],
                "frame_wh": [w_frame, h_frame],
                "crop_wh": [w_crop, h_crop],
            }
            await publisher.publish_face_jpeg(lecture_id, jpeg, metadata=metadata)
            published += 1
        return {"published": published, "faces": len(faces), "ts": ts, "lecture_id": lecture_id, "mode": "faces"}

    async def processor() -> None:
        """Process browser frames: detect → annotate → send back + auto-publish."""
        nonlocal last_auto_publish, last_face_count
        while True:
            async with frame_lock:
                frame = browser_frame
                ts = browser_frame_ts

            if frame is None:
                await asyncio.sleep(0.05)
                continue

            faces = detector.detect(frame)
            current_time = asyncio.get_event_loop().time()

            # Send annotated frame back to browser
            annotated = detector.annotate(frame, faces)
            jpg = camera.encode_jpeg(annotated, settings.jpeg_quality)
            await ws.send_bytes(jpg)

            # Auto-publish face crops to RabbitMQ
            if settings.auto_publish_interval > 0:
                time_since_last = current_time - last_auto_publish
                should_publish = (
                    time_since_last >= settings.auto_publish_interval
                    and len(faces) > 0
                    and (
                        len(faces) != last_face_count
                        or time_since_last >= settings.auto_publish_interval * 1.5
                    )
                )
                if should_publish:
                    try:
                        result = await _publish_faces_from_frame(frame, ts)
                        await ws.send_text(json.dumps({"type": "auto_publish", "data": result}))
                        last_auto_publish = current_time
                        last_face_count = len(faces)
                    except Exception as e:
                        print(f"Auto-publish failed: {e}")

            await asyncio.sleep(0.03)  # ~30fps cap for processing

    async def receiver() -> None:
        """Receive binary JPEG frames and text JSON commands from browser."""
        nonlocal browser_frame, browser_frame_ts
        while True:
            msg = await ws.receive()
            ws_type = msg.get("type", "")

            # Binary message = JPEG frame from browser camera
            if ws_type == "websocket.receive" and "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                arr = np.frombuffer(raw, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    async with frame_lock:
                        browser_frame = decoded
                        browser_frame_ts = time.time()
                continue

            # Text message = JSON command
            if ws_type == "websocket.receive" and "text" in msg and msg["text"]:
                try:
                    payload: dict[str, Any] = json.loads(msg["text"])
                except Exception:
                    continue

                if payload.get("type") == "recognize_now":
                    async with frame_lock:
                        frame = browser_frame
                        ts = browser_frame_ts
                    if frame is None:
                        await ws.send_text(json.dumps({"type": "error", "error": "no_frame_yet", "lecture_id": lecture_id}))
                        continue
                    try:
                        result = await _publish_faces_from_frame(frame, ts)
                        await ws.send_text(json.dumps({"type": "recognize_result", "data": result}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "error", "error": str(e), "lecture_id": lecture_id}))
                continue

            # WebSocket disconnect
            if ws_type == "websocket.disconnect":
                break

    task_p = asyncio.create_task(processor())
    task_r = asyncio.create_task(receiver())
    try:
        await asyncio.gather(task_p, task_r)
    except WebSocketDisconnect:
        pass
    finally:
        task_p.cancel()
        task_r.cancel()


def run() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
