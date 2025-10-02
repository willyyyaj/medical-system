from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Text, DateTime
from sqlalchemy.orm import relationship, backref

from .database import Base


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class PatientDB(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    birthDate = Column(String)
    gender = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    appointments = relationship("AppointmentDB", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("PrescriptionDB", back_populates="patient", cascade="all, delete-orphan")
    tasks = relationship("TaskDB", back_populates="patient", cascade="all, delete-orphan")


class DoctorDB(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    specialty = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppointmentDB(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    appointment_date = Column(String)
    reason = Column(String)
    summary = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    appointment_type = Column(String, nullable=False, default="scheduled", server_default="scheduled")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    patient = relationship("PatientDB", back_populates="appointments")
    doctor = relationship("DoctorDB", backref=backref("appointments", cascade="all, delete-orphan"))
    tasks = relationship("TaskDB", back_populates="appointment", cascade="all, delete-orphan")
    prescriptions = relationship("PrescriptionDB", back_populates="appointment", cascade="all, delete-orphan")


class TaskDB(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    due_date = Column(String)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True, index=True)
    patient = relationship("PatientDB", back_populates="tasks")
    appointment = relationship("AppointmentDB", back_populates="tasks")


class PrescriptionDB(Base):
    __tablename__ = "prescriptions"
    id = Column(Integer, primary_key=True, index=True)
    medication_name = Column(String)
    medication_code = Column(String, nullable=True, index=True)
    dosage = Column(String)
    frequency = Column(String)
    prescribed_on = Column(String)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("PatientDB", back_populates="prescriptions")
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True, index=True)
    appointment = relationship("AppointmentDB", back_populates="prescriptions")





