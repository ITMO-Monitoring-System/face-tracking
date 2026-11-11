import json
import logging
import unittest

from app.logging_utils import JsonFormatter


class LoggingUtilsTest(unittest.TestCase):
    def test_camera_and_broker_credentials_are_masked(self):
        formatter = JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, __file__, 1,
                                   "camera rtsp://user:camera-pass@host/live; amqp://guest:broker-pass@rabbit/", (), None)
        event = json.loads(formatter.format(record))
        self.assertEqual(event["service"], "face_tracking")
        self.assertNotIn("camera-pass", event["message"])
        self.assertNotIn("broker-pass", event["message"])


if __name__ == "__main__":
    unittest.main()
