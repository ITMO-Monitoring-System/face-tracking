import os


class Settings:
    # Camera:
    # - "0" / "1" (как строка) -> локальная вебка
    # - "rtsp://..."          -> RTSP
    # - "none"                -> камера отключена
    camera_source: str = os.getenv("CAMERA_SOURCE") or os.getenv("CAMERA_ID", "0")
    camera_reconnect_interval: float = float(os.getenv("CAMERA_RECONNECT_INTERVAL", "1.0"))

    fps: int = int(os.getenv("FPS", "30"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))

    # Haar cascades
    face_cascade_path: str = os.getenv(
        "FACE_CASCADE_PATH",
        "haarcascades/Haarcascade Frontal Face.xml",
    )

    # RabbitMQ
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5673/")
    exchange_name: str = os.getenv("RABBITMQ_EXCHANGE", "faces")

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
    connect_service_url: str = os.getenv("CONNECT_SERVICE_URL", "http://89.111.170.130:8180/connect")
    connect_service_timeout_seconds: float = float(os.getenv("CONNECT_SERVICE_TIMEOUT_SECONDS", "5.0"))

    connect_service_in_amqp_url: str = os.getenv("CONNECT_SERVICE_IN_AMQP_URL") or rabbitmq_url

    connect_service_threshold: float = float(os.getenv("CONNECT_SERVICE_THRESHOLD", "0.45"))

    connect_service_out_amqp_url: str | None = os.getenv("CONNECT_SERVICE_OUT_AMQP_URL")
    connect_service_out_queue: str | None = os.getenv("CONNECT_SERVICE_OUT_QUEUE")

    lecture_start_ready_timeout_seconds: float = float(os.getenv("LECTURE_START_READY_TIMEOUT_SECONDS", "15.0"))


settings = Settings()
