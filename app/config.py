import json
import os


# Парсим список доступных камер из env один раз при импорте
_camera_sources_raw = os.getenv("CAMERA_SOURCES", "[]")
try:
    _camera_sources_parsed: list = json.loads(_camera_sources_raw)
    if not isinstance(_camera_sources_parsed, list):
        _camera_sources_parsed = []
except Exception:
    _camera_sources_parsed = []


class Settings:
    # Camera:
    # - "0" / "1" (как строка) -> локальная вебка
    # - "rtsp://..."          -> RTSP
    # - "none"                -> камера отключена
    camera_source: str = os.getenv("CAMERA_SOURCE") or os.getenv("CAMERA_ID", "0")
    camera_reconnect_interval: float = float(os.getenv("CAMERA_RECONNECT_INTERVAL", "1.0"))

    # Список доступных камер для выбора с UI.
    # Задаётся через CAMERA_SOURCES как JSON-массив строк.
    # Пример: '["rtsp://cam1:8554/live", "rtsp://cam2:8554/live", "0"]'
    camera_sources: list = _camera_sources_parsed

    fps: int = int(os.getenv("FPS", "30"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))

    # Publish mode:
    # - "faces" -> send cropped faces (default)
    # - "frame" -> send full frame (for testing)
    publish_mode: str = os.getenv("PUBLISH_MODE", "faces").strip().lower()

    # Face crop padding (to include head/around-face context).
    # Values are relative to detected FACE box size.
    face_pad_x_ratio: float = float(os.getenv("FACE_PAD_X_RATIO", "0.35"))
    face_pad_y_ratio: float = float(os.getenv("FACE_PAD_Y_RATIO", "0.35"))
    face_top_extra_ratio: float = float(os.getenv("FACE_TOP_EXTRA_RATIO", "0.60"))
    face_bottom_extra_ratio: float = float(os.getenv("FACE_BOTTOM_EXTRA_RATIO", "0.25"))

    face_crop_make_square: bool = os.getenv("FACE_CROP_MAKE_SQUARE", "true").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }

    # MediaPipe Face Detection
    # Порог уверенности (0.0–1.0): чем выше — тем меньше ложных срабатываний.
    # 0.5 — баланс между чувствительностью и точностью.
    face_min_confidence: float = float(os.getenv("FACE_MIN_CONFIDENCE", "0.5"))

    # model_selection: 0 = short-range (до 2м), 1 = full-range (до 5м, для аудитории)
    face_model_selection: int = int(os.getenv("FACE_MODEL_SELECTION", "1"))

    # RabbitMQ
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5673/")
    exchange_name: str = os.getenv("RABBITMQ_EXCHANGE", "faces")
    auto_publish_interval: float = float(os.getenv("AUTO_PUBLISH_INTERVAL", "10.0"))

    # Per-lecture routing (очередь на лекцию)
    lecture_queue_template: str = os.getenv(
        "RABBITMQ_LECTURE_QUEUE_TEMPLATE",
        "faces.queue.lecture.{lecture_id}",
    )
    lecture_routing_key_template: str = os.getenv(
        "RABBITMQ_LECTURE_ROUTING_KEY_TEMPLATE",
        "lecture.{lecture_id}.face",
    )

    # External connect service (notify on lecture start)
    connect_service_url: str = os.getenv(
        "CONNECT_SERVICE_URL",
        "https://projctviscon.vps.webdock.cloud/recognizing/api/lecture/start",
    )
    connect_service_timeout_seconds: float = float(os.getenv("CONNECT_SERVICE_TIMEOUT_SECONDS", "5.0"))

    connect_service_in_amqp_url: str = os.getenv("CONNECT_SERVICE_IN_AMQP_URL") or rabbitmq_url

    connect_service_threshold: float = float(os.getenv("CONNECT_SERVICE_THRESHOLD", "0.2"))

    connect_service_out_amqp_url: str | None = os.getenv("CONNECT_SERVICE_OUT_AMQP_URL")
    connect_service_out_queue: str | None = os.getenv("CONNECT_SERVICE_OUT_QUEUE")

    lecture_start_ready_timeout_seconds: float = float(os.getenv("LECTURE_START_READY_TIMEOUT_SECONDS", "15.0"))


settings = Settings()
