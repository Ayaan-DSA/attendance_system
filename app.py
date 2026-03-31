from flask import Flask, render_template, Response, jsonify, request
from models import db, Student, Attendance
import recognize_face as rf
import register_face as reg_mod
from train_model import train_model
from datetime import datetime
import os

app = Flask(__name__)

# ── Database config (MySQL via MySQL Workbench) ───────────────────────────────
#  Change these values to match your MySQL setup:
DB_USER     = "Your Username here"
DB_PASSWORD = "YOUR PASSWORD HERE"                    # ← enter your MySQL password here if you have one
DB_HOST     = "localhost"
DB_PORT     = "3306"
DB_NAME     = "attendance_system"   # ← must already exist in MySQL Workbench

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 280,    # prevents MySQL "gone away" errors on idle connections
    "pool_pre_ping": True,
}
db.init_app(app)

with app.app_context():
    db.create_all()   # auto-creates tables inside attendance_system if not already there

# Pass db + app to recognize_face so it can write attendance rows
rf.init_db(app, db, Student, Attendance)

# ── Home ──────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

# ── Camera ────────────────────────────────────────────────────────────────────

@app.route("/start_camera", methods=["POST"])
def start_camera():
    try:
        if not rf.camera_active:
            rf.start_camera()
        return jsonify({"status": "ok", "message": "Camera started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    rf.stop_recognition()
    reg_mod.stop_registration()
    rf.stop_camera()
    return jsonify({"status": "ok"})

@app.route("/video_feed")
def video_feed():
    return Response(
        rf.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ── Registration ──────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    name       = (data.get("name") or "").strip()
    student_id = (data.get("student_id") or "").strip()

    if not name or not student_id:
        return jsonify({"status": "error", "message": "Name and Student ID are required"}), 400
    if not rf.camera_active:
        return jsonify({"status": "error", "message": "Please start the camera first"}), 400
    if reg_mod.registration_running:
        return jsonify({"status": "error", "message": "Registration already running"}), 400

    def get_raw_frame():
        with rf.frame_lock:
            return rf.current_frame.copy() if rf.current_frame is not None else None

    def set_overlay(frame):
        with rf.frame_lock:
            rf.overlay_frame = frame

    def done(count):
        with rf.frame_lock:
            rf.overlay_frame = None
        # Save student to DB (upsert — update name if student_id already exists)
        with app.app_context():
            existing = Student.query.filter_by(student_id=student_id).first()
            if not existing:
                student = Student(name=name, student_id=student_id)
                db.session.add(student)
                db.session.commit()
                print(f"[DB] New student saved: {name} ({student_id})")
            else:
                existing.name = name
                db.session.commit()
                print(f"[DB] Student updated: {name} ({student_id})")
        print(f"[register] Done: {count} images for {name}_{student_id}")

    reg_mod.start_registration_thread(name, student_id, get_raw_frame, set_overlay, done)
    return jsonify({"status": "started", "message": f"Capturing faces for {name} ({student_id})"})

@app.route("/registration_status")
def registration_status():
    return jsonify({
        "running": reg_mod.registration_running,
        "count":   reg_mod.registration_count
    })

@app.route("/stop_register", methods=["POST"])
def stop_register():
    reg_mod.stop_registration()
    return jsonify({"status": "ok"})

# ── Students ──────────────────────────────────────────────────────────────────

@app.route("/students")
def get_students():
    students = Student.query.order_by(Student.registered_at.desc()).all()
    return jsonify([s.to_dict() for s in students])

@app.route("/students/delete", methods=["POST"])
def delete_student():
    data       = request.get_json(force=True)
    student_id = (data.get("student_id") or "").strip()

    if not student_id:
        return jsonify({"status": "error", "message": "student_id required"}), 400

    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    try:
        # Delete attendance records first (FK constraint)
        Attendance.query.filter_by(student_id=student_id).delete()
        db.session.delete(student)
        db.session.commit()
        return jsonify({"status": "ok", "message": f"Deleted {student.name} and their attendance records"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Training ──────────────────────────────────────────────────────────────────

@app.route("/train_model", methods=["POST"])
def train():
    try:
        result = train_model()
        return jsonify({"status": "ok", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Recognition ───────────────────────────────────────────────────────────────

@app.route("/start_recognition", methods=["POST"])
def start_recognition():
    if not rf.camera_active:
        return jsonify({"status": "error", "message": "Please start the camera first"}), 400
    if rf.recognition_running:
        return jsonify({"status": "ok", "message": "Already running"})

    data       = request.get_json(force=True, silent=True) or {}
    class_name = (data.get("class_name") or "General").strip()
    if not class_name:
        return jsonify({"status": "error", "message": "Please enter a class name"}), 400

    rf.attendance_log.clear()
    rf.start_recognition_thread(class_name)
    return jsonify({"status": "ok", "message": f"Recognition started for class: {class_name}"})

@app.route("/stop_recognition", methods=["POST"])
def stop_recognition():
    rf.stop_recognition()
    return jsonify({"status": "ok"})

@app.route("/attendance_log")
def get_attendance_log():
    """Live in-memory log for the current session (shown on main page)."""
    return jsonify(rf.attendance_log)

# ── Attendance records (from DB) ──────────────────────────────────────────────

@app.route("/attendance")
def get_attendance():
    """Return attendance records filtered by date and optionally class."""
    date       = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    class_name = request.args.get("class_name", "")
    query      = Attendance.query.filter_by(date=date)
    if class_name:
        query = query.filter_by(class_name=class_name)
    records = query.order_by(Attendance.time).all()
    return jsonify([r.to_dict() for r in records])

@app.route("/attendance/classes")
def get_classes():
    """Return all distinct class names that have attendance records."""
    rows = db.session.query(Attendance.class_name).distinct().order_by(Attendance.class_name).all()
    return jsonify([r[0] for r in rows])

@app.route("/attendance/dates")
def get_attendance_dates():
    """Return all distinct dates that have attendance records."""
    rows = db.session.query(Attendance.date).distinct().order_by(Attendance.date.desc()).all()
    return jsonify([r[0] for r in rows])

@app.route("/attendance/stats")
def get_attendance_stats():
    """Return summary stats: total students, today's count, total records."""
    today = datetime.now().strftime("%Y-%m-%d")
    total_students  = Student.query.count()
    today_count     = Attendance.query.filter_by(date=today).count()
    total_records   = Attendance.query.count()
    return jsonify({
        "total_students": total_students,
        "today_count":    today_count,
        "total_records":  total_records
    })

@app.route("/camera_status")
def camera_status():
    return jsonify({
        "camera":       rf.camera_active,
        "recognition":  rf.recognition_running,
        "registration": reg_mod.registration_running
    })

if __name__ == "__main__":
    # use_reloader=False is CRITICAL — reloader double-starts threads and breaks camera
    app.run(debug=True, threaded=True, use_reloader=False)