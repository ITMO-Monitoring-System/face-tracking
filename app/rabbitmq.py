from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import base64
import json
from dataclasses import asdict
import uuid

import aio_pika


@dataclass(frozen=True)
class LectureBinding:
    lecture_id: str
    queue_name: str
    routing_key: str
    queue: aio_pika.Queue


class FacePublisher:
    def __init__(
        self,
        url: str,
        exchange_name: str,
        *,
        exchange_type: aio_pika.ExchangeType = aio_pika.ExchangeType.DIRECT,
        lecture_queue_template: str = "faces.queue.lecture.{lecture_id}",
        lecture_routing_key_template: str = "lecture.{lecture_id}.face",
    ):
        self._url = url
        self._exchange_name = exchange_name
        self._exchange_type = exchange_type
        self._lecture_queue_template = lecture_queue_template
        self._lecture_routing_key_template = lecture_routing_key_template

        self._conn: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.abc.AbstractRobustChannel | None = None
        self._exchange: aio_pika.Exchange | None = None

        self._bindings: dict[str, LectureBinding] = {}

        import asyncio
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._conn = await aio_pika.connect_robust(self._url)
        self._channel = await self._conn.channel()
        await self._channel.set_qos(prefetch_count=50)

        self._exchange = await self._channel.declare_exchange(
            self._exchange_name,
            self._exchange_type,
            durable=True,
        )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    def active_lectures(self) -> list[str]:
        return sorted(self._bindings.keys())

    def is_lecture_active(self, lecture_id: str) -> bool:
        return lecture_id in self._bindings

    def _queue_name(self, lecture_id: str) -> str:
        return self._lecture_queue_template.format(lecture_id=lecture_id)

    def _routing_key(self, lecture_id: str) -> str:
        return self._lecture_routing_key_template.format(lecture_id=lecture_id)

    async def start_lecture(
        self,
        lecture_id: str,
        *,
        durable: bool = True,
        auto_delete: bool = False,
        expires_ms: int | None = None,
        message_ttl_ms: int | None = None,
    ) -> LectureBinding:
        if not self._exchange or not self._channel:
            raise RuntimeError("publisher not connected")

        async with self._lock:
            if lecture_id in self._bindings:
                return self._bindings[lecture_id]

            queue_name = self._queue_name(lecture_id)
            routing_key = self._routing_key(lecture_id)

            arguments: dict[str, Any] = {}
            if expires_ms is not None:
                arguments["x-expires"] = int(expires_ms)
            if message_ttl_ms is not None:
                arguments["x-message-ttl"] = int(message_ttl_ms)

            queue = await self._channel.declare_queue(
                queue_name,
                durable=durable,
                auto_delete=auto_delete,
                arguments=arguments or None,
            )
            await queue.bind(self._exchange, routing_key=routing_key)

            binding = LectureBinding(
                lecture_id=lecture_id,
                queue_name=queue_name,
                routing_key=routing_key,
                queue=queue,
            )
            self._bindings[lecture_id] = binding
            return binding

    async def end_lecture(self, lecture_id: str, *, if_unused: bool = False, if_empty: bool = False) -> bool:
        async with self._lock:
            binding = self._bindings.pop(lecture_id, None)
            if not binding:
                return False
            await binding.queue.delete(if_unused=if_unused, if_empty=if_empty)
            return True

    async def publish_face_jpeg(self, lecture_id: str, jpeg_bytes: bytes, metadata: dict | None) -> None:
        if not self._exchange:
            raise RuntimeError("publisher not connected")

        binding = self._bindings.get(lecture_id)
        if not binding:
            raise RuntimeError(f"lecture not started: {lecture_id}")

        # Кодируем JPEG в base64
        image_b64 = base64.b64encode(jpeg_bytes).decode('ascii')

        # Формируем JSON сообщение
        message_data = {
            "request_id": str(uuid.uuid4()),  # или можно использовать другой ID
            "image_b64": image_b64
        }

        # Добавляем метаданные в сообщение, если они есть
        if metadata:
            message_data.update(metadata)

        # Сериализуем в JSON
        json_body = json.dumps(message_data).encode('utf-8')

        msg = aio_pika.Message(
            body=json_body,
            content_type="application/json",  # Меняем content_type
            headers={"lecture_id": lecture_id},  # Оставляем только lecture_id в заголовках
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(msg, routing_key=binding.routing_key)
