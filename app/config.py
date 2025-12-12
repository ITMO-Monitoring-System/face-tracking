import os

class Settings:
    # Camera
    camera_id: int = int(os.getenv("CAMERA_ID", "0"))
    fps: int = int(os.getenv("FPS", "30"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))

    # Haar cascades
    face_cascade_path: str = os.getenv(
        "FACE_CASCADE_PATH",
        "haarcascades/Haarcascade Frontal Face.xml",
    )

    # RabbitMQ (rabbitmq1)
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5673/")
    exchange_name: str = os.getenv("RABBITMQ_EXCHANGE", "faces")
    routing_key: str = os.getenv("RABBITMQ_ROUTING_KEY", "face")
    queue_name: str = os.getenv("RABBITMQ_QUEUE", "faces.queue")

settings = Settings()