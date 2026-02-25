import cv2 as cv
import os
import time
import threading

registration_running = False
registration_count = 0

def _register_loop(name, student_id, get_raw_frame, set_overlay, on_complete):
    global registration_running, registration_count

    haar_cascade = cv.CascadeClassifier("face.xml")
    save_path = os.path.join("dataset", f"{name}_{student_id}")
    os.makedirs(save_path, exist_ok=True)

    count = 0
    registration_count = 0

    while registration_running and count < 100:
        raw = get_raw_frame()
        if raw is None:
            time.sleep(0.05)
            continue

        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        annotated = raw.copy()

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            count += 1
            face_roi_resized = cv.resize(face_roi, (200, 200))
            cv.imwrite(os.path.join(save_path, f"{count}.jpg"), face_roi_resized)
            cv.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.putText(annotated, f"Capturing: {count}/100", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        set_overlay(annotated)
        registration_count = count
        time.sleep(0.03)

    registration_running = False
    if on_complete:
        on_complete(count)


def start_registration_thread(name, student_id, get_raw_frame, set_overlay, on_complete=None):
    global registration_running
    registration_running = True
    t = threading.Thread(
        target=_register_loop,
        args=(name, student_id, get_raw_frame, set_overlay, on_complete),
        daemon=True
    )
    t.start()


def stop_registration():
    global registration_running
    registration_running = False