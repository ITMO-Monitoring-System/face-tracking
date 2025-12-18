from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .camera import CameraWorker
from .config import settings
from .detector import FaceDetector
from .rabbitmq import FacePublisher
from .lecture_manager import LectureManager

app = FastAPI()

detector = FaceDetector(settings.face_cascade_path)
# camera = CameraWorker(settings.camera_source, detector, reconnect_interval=settings.camera_reconnect_interval)

publisher = FacePublisher(
    settings.rabbitmq_url,
    settings.exchange_name,
    lecture_queue_template=settings.lecture_queue_template,
    lecture_routing_key_template=settings.lecture_routing_key_template,
)

lecture_manager = LectureManager(detector, reconnect_interval=settings.camera_reconnect_interval)


def _resolve_source(lecture_id: str, source: str | None) -> str:
    src = (source or settings.camera_source).strip()

    # Для медиамукс используем lecture_123 формат
    if src.startswith("rtsp://mediamtx:8554/lecture"):
        return f"{src}_{lecture_id}"

    return src


@app.on_event("startup")
async def startup() -> None:
    await publisher.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    lecture_manager.stop_all()
    await publisher.close()


class LectureStartRequest(BaseModel):
    durable: bool = True
    auto_delete: bool = False
    expires_ms: int | None = None
    message_ttl_ms: int | None = None
    camera_source: str | None = None


class LectureEndRequest(BaseModel):
    if_unused: bool = False
    if_empty: bool = False


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "camera_source": settings.camera_source}


@app.get("/api/lectures")
async def list_lectures() -> dict:
    return {"active": publisher.active_lectures()}


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

    src = _resolve_source(lecture_id, req.camera_source)
    lecture_manager.start_camera(lecture_id, src)

    return {
        "ok": True,
        "lecture_id": lecture_id,
        "queue": binding.queue_name,
        "routing_key": binding.routing_key,
        "camera_source": src,
    }


@app.post("/api/lectures/{lecture_id}/end")
async def end_lecture(lecture_id: str, req: LectureEndRequest | None = None) -> dict:
    req = req or LectureEndRequest()

    lecture_manager.stop_camera(lecture_id)
    deleted = await publisher.end_lecture(lecture_id, if_unused=req.if_unused, if_empty=req.if_empty)

    return {"ok": True, "lecture_id": lecture_id, "deleted": deleted}


async def publish_current_faces(lecture_id: str) -> dict:
    cam = lecture_manager.get(lecture_id)
    if not cam:
        return {"published": 0, "error": "camera_not_started", "lecture_id": lecture_id}

    frame, _, ts = cam.snapshot()
    if frame is None:
        return {"published": 0, "error": "no_frame_yet", "lecture_id": lecture_id}

    # чтобы не блокировать event loop
    faces = await asyncio.to_thread(detector.detect, frame)
    crops = detector.crop_faces(frame, faces)

    published = 0
    for idx, (bbox, crop) in enumerate(crops):
        jpeg = await asyncio.to_thread(cam.encode_jpeg, crop, settings.jpeg_quality)
        headers = {
            "ts": ts,
            "camera_source": cam.source,
            "idx": idx,
            "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
        }
        await publisher.publish_face_jpeg(lecture_id, jpeg, headers=headers)
        published += 1

    return {"published": published, "faces": len(faces), "ts": ts, "lecture_id": lecture_id}

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
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

    cam = lecture_manager.get(lecture_id)
    if not cam:
        await ws.send_text(json.dumps({"type": "error", "error": "camera_not_started", "lecture_id": lecture_id}))
        await ws.close(code=1008)
        return

    send_interval = 1.0 / max(1, settings.fps)

    async def sender() -> None:
        while True:
            frame, _, _ = cam.snapshot()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            faces = await asyncio.to_thread(detector.detect, frame)
            annotated = detector.annotate(frame, faces)
            jpg = await asyncio.to_thread(cam.encode_jpeg, annotated, settings.jpeg_quality)
            await ws.send_bytes(jpg)
            await asyncio.sleep(send_interval)

    async def receiver() -> None:
        while True:
            msg = await ws.receive_text()
            try:
                payload: dict[str, Any] = json.loads(msg)
            except Exception:
                continue

            if payload.get("type") == "recognize_now":
                result = await publish_current_faces(lecture_id)
                await ws.send_text(json.dumps({"type": "recognize_result", "data": result}))

    task_s = asyncio.create_task(sender())
    task_r = asyncio.create_task(receiver())
    try:
        await asyncio.gather(task_s, task_r)
    except WebSocketDisconnect:
        pass
    finally:
        task_s.cancel()
        task_r.cancel()


def run() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
