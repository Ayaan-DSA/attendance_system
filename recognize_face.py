import cv2 as cv
import pickle
import threading
import queue
import time
from datetime import datetime

# ── Shared state ──────────────────────────────────────────────────────────────
camera_active       = False
current_frame       = None
overlay_frame       = None
frame_lock          = threading.Lock()
recognition_running = False
attendance_log      = []
current_class       = "General"

# Thread-safe queue for DB writes — one dedicated writer thread drains it
_db_queue = queue.Queue()

# Injected by app.py
_flask_app   = None
_db          = None
_Student     = None
_Attendance  = None

def init_db(flask_app, db, Student, Attendance):
    global _flask_app, _db, _Student, _Attendance
    _flask_app  = flask_app
    _db         = db
    _Student    = Student
    _Attendance = Attendance
    # Start the single dedicated DB-writer thread
    t = threading.Thread(target=_db_writer_loop, daemon=True)
    t.start()

# ── DB writer thread (single thread owns all DB sessions) ─────────────────────

def _db_writer_loop():
    """Single thread that drains _db_queue and writes to DB safely."""
    while True:
        item = _db_queue.get()   # blocks until something arrives
        if item is None:
            break
        name, student_id, class_name, date, time_str = item
        try:
            with _flask_app.app_context():
                exists = _Attendance.query.filter_by(
                    student_id=student_id,
                    class_name=class_name,
                    date=date
                ).first()
                if not exists:
                    row = _Attendance(
                        student_id=student_id,
                        class_name=class_name,
                        date=date,
                        time=time_str
                    )
                    _db.session.add(row)
                    _db.session.commit()
                    print(f"[DB] ✓ Saved: {name} ({student_id}) | {class_name} | {date} {time_str}")
                else:
                    print(f"[DB] Already exists: {name} ({student_id}) | {class_name} | {date}")
        except Exception as e:
            print(f"[DB] ✗ ERROR saving attendance for {name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            _db_queue.task_done()

# ── Camera thread ─────────────────────────────────────────────────────────────

def _camera_loop(cam):
    global current_frame, camera_active
    while camera_active:
        ret, frame = cam.read()
        if not ret:
            print("[Camera] Failed to read frame")
            break
        with frame_lock:
            current_frame = frame.copy()
        time.sleep(0.03)
    cam.release()
    with frame_lock:
        current_frame = None
    print("[Camera] Stopped")

def start_camera():
    global camera_active
    if camera_active:
        return
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera — is it connected and not in use?")
    camera_active = True
    t = threading.Thread(target=_camera_loop, args=(cam,), daemon=True)
    t.start()
    print("[Camera] Started")

def stop_camera():
    global camera_active, overlay_frame
    camera_active = False
    with frame_lock:
        overlay_frame = None

# ── MJPEG generator ───────────────────────────────────────────────────────────

def generate_frames():
    while True:
        if not camera_active:
            time.sleep(0.1)
            continue
        with frame_lock:
            src   = overlay_frame if overlay_frame is not None else current_frame
            frame = src.copy() if src is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

# ── Face recognition thread ───────────────────────────────────────────────────

def _recognition_loop():
    global recognition_running, overlay_frame

    print("[Recognition] Starting…")

    # Load model files
    haar_cascade = cv.CascadeClassifier("face.xml")
    if haar_cascade.empty():
        print("[Recognition] ERROR: face.xml not found or empty!")
        recognition_running = False
        return

    if not __import__("os").path.exists("trainer/face_trained.yml"):
        print("[Recognition] ERROR: trainer/face_trained.yml not found — train the model first!")
        recognition_running = False
        return

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trainer/face_trained.yml")

    with open("trainer/people.pkl", "rb") as f:
        people = pickle.load(f)

    print(f"[Recognition] Loaded {len(people)} people: {people}")

    seen_this_session = set()
    today      = datetime.now().strftime("%Y-%m-%d")
    class_name = current_class

    print(f"[Recognition] Class: {class_name} | Date: {today}")

    while recognition_running and camera_active:
        with frame_lock:
            raw = current_frame.copy() if current_frame is not None else None

        if raw is None:
            time.sleep(0.05)
            continue

        gray      = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        faces     = haar_cascade.detectMultiScale(gray, 1.1, 4)
        annotated = raw.copy()

        cv.putText(annotated, f"Class: {class_name}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        for (x, y, w, h) in faces:
            face_roi          = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)

            print(f"[Recognition] Face detected — label={label}, confidence={confidence:.1f}")

            if confidence < 65:
                entry      = people[label]
                parts      = entry.rsplit("_", 1)
                name       = parts[0] if len(parts) == 2 else entry
                student_id = parts[1] if len(parts) == 2 else "N/A"
                display    = f"{name} ({student_id})"
                color      = (0, 220, 0)
            else:
                name, student_id, display, entry = "Unknown", "", "Unknown", None
                color = (0, 0, 220)

            cv.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv.putText(annotated, display, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if name != "Unknown" and entry and entry not in seen_this_session:
                seen_this_session.add(entry)
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[Recognition] ✓ Recognized: {name} ({student_id}) — queuing DB save")
                attendance_log.append({
                    "name": name,
                    "student_id": student_id,
                    "class_name": class_name,
                    "time": now
                })
                _db_queue.put((name, student_id, class_name, today, now))

        with frame_lock:
            overlay_frame = annotated

        time.sleep(0.04)

    with frame_lock:
        overlay_frame = None
    print("[Recognition] Stopped")

def start_recognition_thread(class_name="General"):
    global recognition_running, current_class
    current_class       = class_name.strip() or "General"
    recognition_running = True
    t = threading.Thread(target=_recognition_loop, daemon=True)
    t.start()

def stop_recognition():
    global recognition_running
    recognition_running = False