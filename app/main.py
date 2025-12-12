from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .camera import CameraWorker
from .config import settings
from .detector import FaceDetector
from .rabbitmq import FacePublisher


app = FastAPI()

detector = FaceDetector(settings.face_cascade_path)
camera = CameraWorker(settings.camera_id, detector)
publisher = FacePublisher(
    settings.rabbitmq_url,
    settings.exchange_name,
    settings.queue_name,
    settings.routing_key,
)


@app.on_event("startup")
async def startup() -> None:
    camera.start()
    await publisher.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    camera.stop()
    await publisher.close()


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


async def publish_current_faces() -> dict:
    frame, faces, ts = camera.snapshot()
    if frame is None:
        return {"published": 0, "error": "no_frame_yet"}

    crops = detector.crop_faces(frame, faces)
    published = 0
    for idx, (bbox, crop) in enumerate(crops):
        jpeg = camera.encode_jpeg(crop, settings.jpeg_quality)
        headers = {
            "ts": ts,
            "camera_id": settings.camera_id,
            "idx": idx,
            "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
        }
        await publisher.publish_face_jpeg(jpeg, headers=headers)
        published += 1

    return {"published": published, "faces": len(faces), "ts": ts}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()

    send_interval = 1.0 / max(1, settings.fps)

    async def sender() -> None:
        while True:
            frame, faces, _ = camera.snapshot()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

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
                result = await publish_current_faces()
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)