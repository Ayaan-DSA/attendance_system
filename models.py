from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Student(db.Model):
    __tablename__ = "students"

    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100), nullable=False)
    student_id   = db.Column(db.String(50), unique=True, nullable=False)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # one student → many attendance rows
    records = db.relationship("Attendance", backref="student", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "student_id": self.student_id,
            "registered_at": self.registered_at.strftime("%Y-%m-%d %H:%M:%S")
        }


class Attendance(db.Model):
    __tablename__ = "attendance"

    id         = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), db.ForeignKey("students.student_id"), nullable=False)
    class_name = db.Column(db.String(100), nullable=False, default="General")
    date       = db.Column(db.String(10),  nullable=False)   # "YYYY-MM-DD"
    time       = db.Column(db.String(8),   nullable=False)   # "HH:MM:SS"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.student.name if self.student else "Unknown",
            "student_id": self.student_id,
            "class_name": self.class_name,
            "date": self.date,
            "time": self.time
        }