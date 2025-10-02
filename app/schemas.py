from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class TranscriptData(BaseModel):
    text: str


class UserCredentials(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class PatientBase(BaseModel):
    name: str
    birthDate: str
    gender: str


class PatientCreate(PatientBase):
    credentials: UserCredentials


class Patient(BaseModel):
    id: int
    user_id: int
    name: str
    birthDate: str
    gender: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DoctorBase(BaseModel):
    name: str
    specialty: str


class DoctorCreate(DoctorBase):
    credentials: UserCredentials


class Doctor(BaseModel):
    id: int
    user_id: int
    name: str
    specialty: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskBase(BaseModel):
    description: str
    due_date: str


class TaskCreate(TaskBase):
    appointment_id: Optional[int] = None


class TaskUpdate(BaseModel):
    is_completed: bool


class Task(TaskBase):
    id: int
    is_completed: bool
    created_at: datetime
    appointment_id: Optional[int] = None

    class Config:
        from_attributes = True


class AppointmentBase(BaseModel):
    appointment_date: str
    reason: str
    patient_id: int


class AppointmentCreate(AppointmentBase):
    pass


class WalkInAppointmentCreate(BaseModel):
    patient_id: int
    reason: str


class Appointment(BaseModel):
    id: int
    doctor_id: int
    appointment_type: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AppointmentForDoctor(BaseModel):
    id: int
    appointment_date: str
    reason: str
    patient: Patient
    appointment_type: str
    created_at: datetime
    tasks: List[Task] = []
    summary: Optional[str] = None

    class Config:
        from_attributes = True


class DashboardData(BaseModel):
    next_appointment: Optional[Appointment] = None
    pending_tasks: List[Task] = []


class PrescriptionBase(BaseModel):
    medication_name: str
    medication_code: Optional[str] = None
    dosage: str
    frequency: str
    patient_id: int
    appointment_id: Optional[int] = None
    instructions: Optional[str] = None


class PrescriptionCreate(PrescriptionBase):
    pass


class Prescription(PrescriptionCreate):
    id: int
    prescribed_on: str
    doctor_id: int
    created_at: datetime
    appointment_id: Optional[int] = None
    medication_code: Optional[str] = None

    class Config:
        from_attributes = True


class TaskForAppointmentDetail(BaseModel):
    description: str
    is_completed: bool

    class Config:
        from_attributes = True


class DoctorForAppointmentDetail(BaseModel):
    name: str
    specialty: str

    class Config:
        from_attributes = True


class AppointmentDetail(BaseModel):
    id: int
    appointment_date: str
    reason: str
    doctor: DoctorForAppointmentDetail
    tasks: List[TaskForAppointmentDetail] = []
    summary: Optional[str] = "AI 摘要功能開發中..."

    class Config:
        from_attributes = True


class AppointmentDetailForPatient(BaseModel):
    id: int
    appointment_date: str
    reason: str
    doctor: DoctorForAppointmentDetail
    tasks: List[Task] = []
    summary: Optional[str] = None
    tags: Optional[str] = None
    appointment_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class SummaryUpdate(BaseModel):
    summary: str


class QuestionItem(BaseModel):
    question: str
    record_date: datetime






