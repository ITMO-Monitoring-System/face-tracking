from __future__ import annotations

import aio_pika


class FacePublisher:
    def __init__(self, url: str, exchange_name: str, queue_name: str, routing_key: str):
        self._url = url
        self._exchange_name = exchange_name
        self._queue_name = queue_name
        self._routing_key = routing_key

        self._conn: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.abc.AbstractRobustChannel | None = None
        self._exchange: aio_pika.Exchange | None = None

    async def connect(self) -> None:
        self._conn = await aio_pika.connect_robust(self._url)
        self._channel = await self._conn.channel()
        await self._channel.set_qos(prefetch_count=50)

        self._exchange = await self._channel.declare_exchange(
            self._exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await self._channel.declare_queue(self._queue_name, durable=True)
        await queue.bind(self._exchange, routing_key=self._routing_key)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def publish_face_jpeg(self, jpeg_bytes: bytes, headers: dict) -> None:
        if not self._exchange:
            raise RuntimeError("publisher not connected")

        msg = aio_pika.Message(
            body=jpeg_bytes,
            content_type="image/jpeg",
            headers=headers,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(msg, routing_key=self._routing_key)