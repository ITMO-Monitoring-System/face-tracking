"""Control the host lecture archive through its private Unix socket."""

import asyncio
import json


async def control(socket_path: str | None, action: str, lecture_id: str) -> dict | None:
    if not socket_path:
        return None
    if action not in {"start", "stop"}:
        raise ValueError("invalid archive action")
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(socket_path), timeout=5)
    try:
        writer.write((json.dumps({"action": action, "lecture_id": lecture_id}) + "\n").encode())
        await asyncio.wait_for(writer.drain(), timeout=5)
        raw = await asyncio.wait_for(reader.readline(), timeout=15)
        result = json.loads(raw)
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError(result.get("error", "archive control failed") if isinstance(result, dict)
                               else "archive control failed")
        return result
    finally:
        writer.close()
        await writer.wait_closed()
