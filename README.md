# face-tracking
face tracking


Как запускать RTSP(стрим вебки) с ноута на сервер

На сервере
```bash
docker compose up -d --build
```


Открыть:
```
http://89.111.170.130:8000/
```


На ноуте (пушим вебку в RTSP на сервер)

Пример для macOS:

Посмотреть устройства:
```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

Стрим на сервер:
```
ffmpeg \
  -f avfoundation \
  -framerate 30 \
  -video_size 1280x720 \
  -pix_fmt uyvy422 \
  -i "0" \
  -vf "format=yuv420p" \
  -c:v h264_videotoolbox \
  -b:v 1000k \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://89.111.170.130:8554/lecture
```

После этого серверный контейнер начнёт видеть кадры (если стрим пропадёт — он будет переподключаться сам).
