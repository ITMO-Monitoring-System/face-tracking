import json
import logging
import re
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        message = re.sub(r"(?i)\b(amqps?|postgres(?:ql)?|redis|rtsp|https?)://[^\s/@:]+:[^\s/@]+@", r"\1://[REDACTED]@", message)
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict) or "event" not in payload:
                raise ValueError
        except ValueError:
            payload = {"event": "legacy_log", "message": message}
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        payload.setdefault("level", record.levelname.lower())
        payload.setdefault("service", "face_tracking")
        return json.dumps(payload, ensure_ascii=False)


def setup():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())


def event(name, lecture_id=None, **fields):
    payload = {"event": name, "lecture_id": lecture_id, **fields}
    logging.getLogger("face_tracking").info(json.dumps(payload, ensure_ascii=False))
