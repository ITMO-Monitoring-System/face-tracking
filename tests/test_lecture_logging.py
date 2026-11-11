import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.lecture_logging import control


class LectureLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_socket_start_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "collector.sock")
            requests = []

            async def handle(reader, writer):
                request = json.loads(await reader.readline())
                requests.append(request)
                writer.write((json.dumps({"ok": True, "session_id": "archive-1",
                                          "state": "active" if request["action"] == "start" else "finalizing"}) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path)
            try:
                started = await control(path, "start", "12")
                stopped = await control(path, "stop", "12")
            finally:
                server.close()
                await server.wait_closed()
            self.assertEqual(started["session_id"], stopped["session_id"])
            self.assertEqual(stopped["state"], "finalizing")
            self.assertEqual(requests, [{"action": "start", "lecture_id": "12"},
                                        {"action": "stop", "lecture_id": "12"}])

    async def test_unavailable_collector_fails_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                await control(str(Path(directory) / "missing.sock"), "start", "12")


if __name__ == "__main__":
    unittest.main()
