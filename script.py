import cv2
import os
import time

# --- Configuration ---
OUTPUT_DIR = "data/buckets"
FACE_CASCADE_PATH = "haarcascades/Haarcascade Frontal Face.xml"
EYES_CASCADE_PATH = "haarcascades/Haarcascade Eye.xml"
CAMERA_ID = 0  # 0 - built-in, 1 - external
SAVE_INTERVAL = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dataset(img, folder, filename):
    target_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, filename)
    cv2.imwrite(path, img)
    return path


def draw_boundaries(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 6), cv2.FONT_ITALIC, 0.6, color, 1, cv2.LINE_AA)
        coords.append((x, y, w, h))
    return coords, img


def detect_faces(img, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    faces, annotated = draw_boundaries(img, faceCascade, 1.1, 5, color["blue"], "face")
    return faces, annotated


def save_detected_faces(img, faces, bucket_id):
    saved = 0
    timestamp = int(time.time())

    for i, (x, y, w, h) in enumerate(faces):
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi_img = img[y1:y2, x1:x2]
        folder = f"bucket-{bucket_id}"
        filename = f"face-{bucket_id}-{i}-{timestamp}.jpg"

        generate_dataset(roi_img, folder, filename)
        saved += 1

    return saved


if __name__ == '__main__':
    if not os.path.exists(FACE_CASCADE_PATH):
        print(f"Warning: face cascade not found at '{FACE_CASCADE_PATH}'. Please check path.")

    faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    video_capture = cv2.VideoCapture(CAMERA_ID)

    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open camera id={CAMERA_ID}")

    frame_idx = 0
    last_save_time = time.time()
    bucket_counter = 0
    saved_faces_from_last_detection = []

    print("Press 's' to save faces from current frame manually, 'q' to quit.")
    print(f"Auto-saving faces every {SAVE_INTERVAL} seconds...")

    while True:
        ret, img = video_capture.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            break

        current_time = time.time()

        faces, annotated = detect_faces(img, faceCascade)

        saved_faces_from_last_detection = faces

        if current_time - last_save_time >= SAVE_INTERVAL:
            if saved_faces_from_last_detection:
                saved_count = save_detected_faces(
                    img.copy(),
                    saved_faces_from_last_detection,
                    bucket_counter
                )
                print(f"[AUTO] Saved {saved_count} faces at {time.strftime('%H:%M:%S')} into bucket-{bucket_counter}")
            else:
                print(f"[AUTO] No faces detected at {time.strftime('%H:%M:%S')}")

            last_save_time = current_time
            bucket_counter += 1

        time_remaining = SAVE_INTERVAL - (current_time - last_save_time)
        cv2.putText(annotated, f"Next save in: {int(time_remaining)}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("face detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if faces:
                saved_count = save_detected_faces(img.copy(), faces, f"manual-{bucket_counter}")
                print(f"[MANUAL] Saved {saved_count} faces at {time.strftime('%H:%M:%S')}")
            else:
                print("[MANUAL] No faces to save")

        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()