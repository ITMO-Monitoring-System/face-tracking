import cv2
import os
import time

# --- Configuration ---
OUTPUT_DIR = "data/buckets"
FACE_CASCADE_PATH = "haarcascades/Haarcascade Frontal Face.xml"
EYES_CASCADE_PATH = "haarcascades/Haarcascade Eye.xml"
CAMERA_ID = 0  # 0 - built-in, 1 - external


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


def detect_and_save(img, faceCascade, frame_idx):
    color = {"blue": (255, 0, 0),
             "red": (0, 0, 255),
             "green": (0, 255, 0)
             }

    faces, annotated = draw_boundaries(img, faceCascade, 1.1, 5, color["blue"], "face")

    saved = 0
    timestamp = int(time.time())

    for i, (x, y, w, h) in enumerate(faces):
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        roi_img = img[y1:y2, x1:x2]

        folder = f"backet-{frame_idx}"
        filename = f"face-{frame_idx}-{i}-{timestamp}.jpg"

        saved_path = generate_dataset(roi_img, folder, filename)
        saved += 1
        cv2.putText(annotated, f"#{i}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color["red"], 2)

    return annotated, saved


if __name__ == '__main__':
    if not os.path.exists(FACE_CASCADE_PATH):
        print(f"Warning: face cascade not found at '{FACE_CASCADE_PATH}'. Please check path.")

    faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    video_capture = cv2.VideoCapture(CAMERA_ID)
    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open camera id={CAMERA_ID}")

    frame_idx = 0
    last_save_time = time.time()

    print("Press 's' to save faces from current frame manually, 'q' to quit.")

    while True:
        ret, img = video_capture.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            break

        # TODO: разобраться с сохранением бакетов с лицами каждую минуту. Сейчас сохраняет каждую итерацию
        # current_time = time.time()
        # if current_time - last_save_time >= 60:
        #     _, saved = detect_and_save(img.copy(), faceCascade, frame_idx)
        #     print(f"[AUTO] Saved {saved} faces from frame {frame_idx} into {OUTPUT_DIR}/frame-{frame_idx}")
        #     last_save_time = current_time

        annotated, _ = detect_and_save(img.copy(), faceCascade, frame_idx)
        cv2.imshow("face detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # elif key == ord('s'):
        #     _, saved = detect_and_save(img.copy(), faceCascade, frame_idx)
        #     print(f"Saved {saved} faces from frame {frame_idx} into {OUTPUT_DIR}/frame-{frame_idx}")

        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()
