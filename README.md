# face-tracking
face tracking


Как запускать стрим вебки с ноута на фронт

На сервере
```bash
docker compose up -d --build
```


Открыть:
```
http://89.111.170.130/
```


На ноуте:

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
  -preset ultrafast \
  -tune zerolatency \
  -g 30 \
  -f rtsp \
  -rtsp_transport tcp \
  -muxdelay 0.1 \
  "rtsp://89.111.170.130:8554/lecture"
```

После этого серверный контейнер начнёт видеть кадры (если стрим пропадёт — он будет переподключаться сам).
