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
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0" \
  -vcodec libx264 -preset veryfast -tune zerolatency \
  -f rtsp rtsp://89.111.170.130:8554/lecture
```

После этого серверный контейнер начнёт видеть кадры (если стрим пропадёт — он будет переподключаться сам).
