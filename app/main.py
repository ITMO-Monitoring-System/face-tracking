from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import httpx
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .camera import CameraWorker
from .config import settings
from .detector import FaceDetector
from .rabbitmq import FacePublisher

app = FastAPI()

detector = FaceDetector(settings.face_cascade_path)
camera = CameraWorker(settings.camera_source, detector, reconnect_interval=settings.camera_reconnect_interval)

publisher = FacePublisher(
    settings.rabbitmq_url,
    settings.exchange_name,
    lecture_queue_template=settings.lecture_queue_template,
    lecture_routing_key_template=settings.lecture_routing_key_template,
)


class LectureStartRequest(BaseModel):
    durable: bool = True
    auto_delete: bool = False
    expires_ms: int | None = None
    message_ttl_ms: int | None = None


class LectureEndRequest(BaseModel):
    if_unused: bool = False
    if_empty: bool = False


@app.on_event("startup")
async def startup() -> None:
    camera.start()
    await publisher.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    camera.stop()
    await publisher.close()


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "camera_source": settings.camera_source}


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

    try:
        frame, ts = await _wait_for_camera_frame(timeout_seconds=settings.lecture_start_ready_timeout_seconds)
        jpeg = camera.encode_jpeg(frame, settings.jpeg_quality)
        await publisher.publish_face_jpeg(
            lecture_id,
            jpeg,
            metadata={
                "type": "probe",
                "ts": ts,
                "camera_source": settings.camera_source,
                "lecture_id": lecture_id,
            },
        )
    except Exception as e:
        await publisher.end_lecture(lecture_id)
        raise HTTPException(status_code=503, detail=f"lecture_not_ready: {e}")

    try:
        async with httpx.AsyncClient(timeout=settings.connect_service_timeout_seconds) as client:
            resp = await client.post(
                settings.connect_service_url,
                json={
                    "lecture_id": lecture_id,
                    "in_amqp_url": settings.connect_service_in_amqp_url,
                    "in_queue": binding.queue_name,
                    "threshold": settings.connect_service_threshold,
                },
            )
            if resp.is_error:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"connect_service_error: HTTP {resp.status_code}: {resp.text}"
                    ),
                )
    except HTTPException:
        await publisher.end_lecture(lecture_id)
        raise
    except Exception as e:
        await publisher.end_lecture(lecture_id)
        raise HTTPException(status_code=502, detail=f"connect_service_error: {e}")
    return {"ok": True, "lecture_id": lecture_id, "queue": binding.queue_name, "routing_key": binding.routing_key}


@app.post("/api/lectures/{lecture_id}/end")
async def end_lecture(lecture_id: str, req: LectureEndRequest | None = None) -> dict:
    req = req or LectureEndRequest()
    deleted = await publisher.end_lecture(lecture_id, if_unused=req.if_unused, if_empty=req.if_empty)
    return {"ok": True, "lecture_id": lecture_id, "deleted": deleted}


async def publish_current_faces(lecture_id: str) -> dict:
    frame, _, ts = camera.snapshot()
    if frame is None:
        return {"published": 0, "error": "no_frame_yet", "lecture_id": lecture_id}

    faces = detector.detect(frame)
    crops = detector.crop_faces(frame, faces)
    published = 0

    for idx, (bbox, crop) in enumerate(crops):
        jpeg = camera.encode_jpeg(crop, settings.jpeg_quality)

        # Формируем метаданные для JSON сообщения
        metadata = {
            "ts": ts,
            "camera_source": settings.camera_source,
            "idx": idx,
            "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
            "lecture_id": lecture_id  # Добавляем lecture_id в тело сообщения
        }

        await publisher.publish_face_jpeg(lecture_id, jpeg, metadata=metadata)
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

    send_interval = 1.0 / max(1, settings.fps)

    async def sender() -> None:
        while True:
            frame, _, _ = camera.snapshot()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            faces = detector.detect(frame)

            annotated = detector.annotate(frame, faces)
            jpg = camera.encode_jpeg(annotated, settings.jpeg_quality)
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
                try:
                    result = await publish_current_faces(lecture_id)
                    await ws.send_text(json.dumps({"type": "recognize_result", "data": result}))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "error": str(e), "lecture_id": lecture_id}))

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
