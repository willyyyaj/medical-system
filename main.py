# -*- coding: utf-8 -*-
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import shutil
import os

# --- Google Gemini AI 相關匯入 ---
import google.generativeai as genai

# --- Whisper ASR 模型相關匯入 ---


# --- JWT & 密碼處理相關匯入 ---
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# --- SQLAlchemy 相關匯入 ---
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship

# --------------------------------------------------------------------------
# 0. 設定日誌 (Logging)
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 設定您的 Google Gemini API 金鑰 ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 環境變數未設定")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logging.info("Google Gemini AI SDK 設定成功。")
except Exception as e:
    gemini_model = None
    logging.error(f"無法設定 Google Gemini AI SDK: {e}")


# --- 載入 Whisper 模型 ---

# --------------------------------------------------------------------------
# 1. 認證與安全設定
# --------------------------------------------------------------------------
SECRET_KEY = "請在此處更換為你的超級秘密金鑰"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --------------------------------------------------------------------------
# 2. 資料庫設定
# --------------------------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./medical_system_final_v2.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------------------------------------------------------------
# 3. SQLAlchemy ORM 模型 (維持不變)
# --------------------------------------------------------------------------
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
    appointments = relationship("AppointmentDB", back_populates="doctor")

class AppointmentDB(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    appointment_date = Column(String)
    reason = Column(String)
    summary = Column(Text, nullable=True) 
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    patient = relationship("PatientDB", back_populates="appointments")
    doctor = relationship("DoctorDB", back_populates="appointments")
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
    dosage = Column(String)
    frequency = Column(String)
    prescribed_on = Column(String)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("PatientDB", back_populates="prescriptions")
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True, index=True)
    appointment = relationship("AppointmentDB", back_populates="prescriptions")

# --------------------------------------------------------------------------
# 4. Pydantic 模型 (維持不變)
# --------------------------------------------------------------------------
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

class Patient(PatientBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class DoctorBase(BaseModel):
    name: str
    specialty: str

class DoctorCreate(DoctorBase):
    credentials: UserCredentials

class Doctor(DoctorBase):
    id: int
    user_id: int
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
    doctor_id: int

class AppointmentCreate(AppointmentBase):
    pass

class Appointment(AppointmentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class AppointmentForDoctor(BaseModel):
    id: int
    appointment_date: str
    reason: str
    patient: Patient
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
    dosage: str
    frequency: str
    patient_id: int
    appointment_id: Optional[int] = None

class PrescriptionCreate(PrescriptionBase):
    pass

class Prescription(PrescriptionCreate):
    id: int
    prescribed_on: str
    doctor_id: int
    created_at: datetime
    appointment_id: Optional[int] = None
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
    class Config:
        from_attributes = True

class SummaryUpdate(BaseModel):
    summary: str

# --------------------------------------------------------------------------
# 5. WebSocket 連線管理器 (維持不變)
# --------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logging.info(f"WebSocket: 使用者 {user_id} 連線成功。")

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logging.info(f"WebSocket: 使用者 {user_id} 已離線。")

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# --------------------------------------------------------------------------
# 6. FastAPI 應用與資料庫初始化 (維持不變)
# --------------------------------------------------------------------------
app = FastAPI(title="智慧醫療資訊系統 API (V2 - 修正版)")
origins = [
    "http://localhost",
    "http://localhost:8888",
    "http://localhost:8889",
    "http://127.0.0.1:8888",
    "http://127.0.0.1:8889",
    "null"
     "https://jovial-swan-576e90.netlify.app"
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine, checkfirst=True)

# --------------------------------------------------------------------------
# 7. 依賴注入與核心認證函式 (維持不變)
# --------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="無法驗證憑證", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

# --------------------------------------------------------------------------
# 8. API 端點 (已修改)
# --------------------------------------------------------------------------

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="不正確的帳號或密碼")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/summarize", tags=["AI"])
async def summarize_text(transcript_data: TranscriptData, current_user: User = Depends(get_current_user)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini 模型未能成功載入，請檢查伺服器日誌。")
    
    # --- [修改] ---
    # 在 Prompt 中明確要求使用繁體中文
    prompt = f"""
    角色：你是一位專業、資深的醫療紀錄專家。
    任務：請將以下提供的「醫病對話逐字稿」轉換成一份標準、嚴謹、客觀的 SOAP Note (病歷摘要)。

    SOAP 格式說明：
    - S (Subjective/主觀陳述): 病患或家屬口頭描述的症狀、感受、病史。請直接引用或客觀總結病患的說法。
    - O (Objective/客觀檢查): 醫生觀察到的生命徵象、檢查數據、理學檢查結果。如果逐字稿中沒有明確數據，請註明「(根據逐字稿，無客觀檢查數據)」。
    - A (Assessment/評估): 醫生根據 S 和 O 所做出的初步診斷或鑑別診斷。
    - P (Plan/計畫): 醫生提出的治療方案、藥物調整、建議的檢查、衛教內容或下次追蹤計畫。

    重要規則：
    1.  **嚴格遵循格式**：必須包含 S, O, A, P 四個部分，每個部分都要有明確的標題。
    2.  **客觀中立**：不要添加任何逐字稿中沒有的個人猜測或評論。
    3.  **簡潔專業**：使用精確的醫療術語，避免口語化表達。
    4.  **專注於醫療資訊**：忽略對話中的閒聊或與病情無關的內容。
    5.  **使用繁體中文**：所有摘要內容必須以繁體中文撰寫。

    醫病對話逐字稿：
    ---
    {transcript_data.text}
    ---

    請生成 SOAP 摘要：
    """
    
    try:
        response = await gemini_model.generate_content_async(prompt)
        summary_text = response.text
        
        logging.info(f"Gemini summary generated for user {current_user.username}")
        return {"summary": summary_text.strip()}

    except Exception as e:
        logging.error(f"Summarization failed with Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"生成摘要失敗: {e}")

@app.get("/dashboard", response_model=DashboardData, tags=["Dashboard"])
def get_dashboard_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="僅限病患存取")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    next_appointment = db.query(AppointmentDB).filter(AppointmentDB.patient_id == patient_profile.id).order_by(AppointmentDB.appointment_date.desc()).first()
    pending_tasks = db.query(TaskDB).filter(TaskDB.patient_id == patient_profile.id, TaskDB.is_completed == False).all()
    return DashboardData(next_appointment=next_appointment, pending_tasks=pending_tasks)

@app.post("/patients/", response_model=Patient, tags=["Patients"])
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == patient.credentials.username).first():
        raise HTTPException(status_code=400, detail="此帳號已被註冊")
    hashed_password = get_password_hash(patient.credentials.password)
    new_user = UserDB(username=patient.credentials.username, hashed_password=hashed_password, role="Patient")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db_patient = PatientDB(**patient.dict(exclude={'credentials'}), user_id=new_user.id)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients/me", response_model=Patient, tags=["Patients"])
def read_patient_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="僅限病患本人操作")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    return patient_profile

@app.get("/patients/me/appointments", response_model=List[AppointmentDetailForPatient], tags=["Patients"])
def read_my_appointments(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="權限不足，僅限病患本人操作")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    return patient_profile.appointments

@app.get("/patients/", response_model=List[Patient], tags=["Patients"])
def read_all_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    return db.query(PatientDB).all()

@app.get("/patients/me/prescriptions", response_model=List[Prescription], tags=["Patients"])
def read_my_prescriptions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="權限不足，僅限病患本人操作")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    return patient_profile.prescriptions

@app.post("/doctors/", response_model=Doctor, tags=["Doctors"])
def create_doctor(doctor: DoctorCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == doctor.credentials.username).first():
        raise HTTPException(status_code=400, detail="此帳號已被註冊")
    hashed_password = get_password_hash(doctor.credentials.password)
    new_user = UserDB(username=doctor.credentials.username, hashed_password=hashed_password, role="Doctor")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db_doctor = DoctorDB(**doctor.dict(exclude={'credentials'}), user_id=new_user.id)
    db.add(db_doctor)
    db.commit()
    db.refresh(db_doctor)
    return db_doctor

@app.get("/doctors/", response_model=List[Doctor], tags=["Doctors"])
def read_doctors(db: Session = Depends(get_db)):
    return db.query(DoctorDB).all()

@app.get("/doctors/me", response_model=Doctor, tags=["Doctors"])
def read_doctor_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生本人操作")
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")
    return doctor_profile

@app.get("/doctors/me/appointments", response_model=List[AppointmentForDoctor], tags=["Doctors"])
def read_doctor_appointments(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生本人操作")
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")
    return doctor_profile.appointments

@app.get("/doctors/me/patients", response_model=List[Patient], tags=["Doctors"], summary="獲取目前登入醫生的病患列表")
def read_doctor_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生本人操作")
    
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")

    patient_ids_query = db.query(AppointmentDB.patient_id).filter(AppointmentDB.doctor_id == doctor_profile.id).distinct()
    
    patient_ids = [p_id for p_id, in patient_ids_query.all()]

    if not patient_ids:
        return []

    patients = db.query(PatientDB).filter(PatientDB.id.in_(patient_ids)).all()
    
    return patients

@app.post("/tasks/", response_model=Task, tags=["Tasks"], summary="建立一個通用任務(舊版)")
def create_task(task: TaskCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="只有病患可以建立任務")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料，無法建立任務")
    if task.appointment_id:
        appointment = db.query(AppointmentDB).filter(AppointmentDB.id == task.appointment_id, AppointmentDB.patient_id == patient_profile.id).first()
        if not appointment:
            raise HTTPException(status_code=404, detail="找不到指定的看診紀錄，或該紀錄不屬於您")
    db_task = TaskDB(description=task.description, due_date=task.due_date, appointment_id=task.appointment_id, patient_id=patient_profile.id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.post("/appointments/{appointment_id}/tasks", response_model=Task, tags=["Tasks"], summary="為特定看診建立準備任務(推薦)")
def create_task_for_appointment(appointment_id: int, task_data: TaskBase, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="只有病患可以建立任務")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id, AppointmentDB.patient_id == patient_profile.id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="找不到指定的看診紀錄，或該紀錄不屬於您")
    db_task = TaskDB(**task_data.dict(), patient_id=patient_profile.id, appointment_id=appointment.id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.put("/tasks/{task_id}", response_model=Task, tags=["Tasks"])
def update_task_status(task_id: int, task_update: TaskUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="只有病患可以修改任務")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    db_task = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="找不到該任務")
    if db_task.patient_id != patient_profile.id:
        raise HTTPException(status_code=403, detail="權限不足，無法修改他人任務")
    db_task.is_completed = task_update.is_completed
    db.commit()
    db.refresh(db_task)
    return db_task

@app.post("/appointments/", response_model=Appointment, tags=["Appointments"])
def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    if not db.query(PatientDB).filter(PatientDB.id == appointment.patient_id).first():
        raise HTTPException(status_code=404, detail="找不到病患")
    if not db.query(DoctorDB).filter(DoctorDB.id == appointment.doctor_id).first():
        raise HTTPException(status_code=404, detail="找不到醫生")
    db_appointment = AppointmentDB(**appointment.dict())
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@app.delete("/appointments/{appointment_id}", status_code=204, tags=["Appointments"])
def delete_appointment(appointment_id: int, db: Session = Depends(get_db)):
    db_appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="找不到該看診紀錄")
    db.delete(db_appointment)
    db.commit()
    return Response(status_code=204)

@app.get("/appointments/{appointment_id}/summary", response_model=AppointmentDetail, tags=["Appointments"], summary="獲取單一看診的詳細摘要")
def get_appointment_summary(appointment_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="找不到該看診紀錄")

    if current_user.role == "Patient":
        patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
        if not patient_profile or appointment.patient_id != patient_profile.id:
            raise HTTPException(status_code=403, detail="權限不足，無法查看此看診紀錄")
    
    if not appointment.summary:
        appointment.summary = "醫生尚未批准或撰寫本次看診的摘要。"

    return appointment

@app.post("/appointments/{appointment_id}/summary", status_code=200, tags=["Appointments"], summary="批准並發送摘要")
async def approve_and_send_summary(appointment_id: int, summary_data: SummaryUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")

    appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="找不到該看診紀錄")
    
    if appointment.doctor_id != doctor_profile.id:
        raise HTTPException(status_code=403, detail="權限不足，無法修改非自己的看診紀錄")

    appointment.summary = summary_data.summary
    db.commit()
    db.refresh(appointment)

    patient_user_id = appointment.patient.user_id
    if patient_user_id:
        notification_message = json.dumps({
            "type": "new_summary",
            "data": {
                "appointment_date": appointment.appointment_date,
                "doctor_name": doctor_profile.name
            }
        })
        await manager.send_personal_message(notification_message, patient_user_id)

    return {"message": "摘要已成功儲存並發送通知"}


@app.get("/patients/{patient_id}/prescriptions", response_model=List[Prescription], tags=["Prescriptions"])
def get_patient_prescriptions(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    patient = db.query(PatientDB).filter(PatientDB.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="找不到該病患")
    return patient.prescriptions

@app.post("/prescriptions/", response_model=Prescription, tags=["Prescriptions"])
async def create_prescription(prescription: PrescriptionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")
    patient_profile = db.query(PatientDB).filter(PatientDB.id == prescription.patient_id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到該病患")
    
    appointment_date = datetime.now().strftime("%Y-%m-%d")
    if prescription.appointment_id:
        appointment = db.query(AppointmentDB).filter(AppointmentDB.id == prescription.appointment_id).first()
        if not appointment:
            raise HTTPException(status_code=404, detail="找不到對應的看診紀錄")
        appointment_date = appointment.appointment_date

    db_prescription = PrescriptionDB(
        **prescription.dict(), 
        doctor_id=doctor_profile.id, 
        prescribed_on=appointment_date
    )
    db.add(db_prescription)
    db.commit()
    db.refresh(db_prescription)
    
    notification_message = json.dumps({ "type": "new_prescription", "data": { "medication_name": db_prescription.medication_name, "doctor_name": doctor_profile.name } })
    if patient_profile.user_id:
        await manager.send_personal_message(notification_message, patient_profile.user_id)
    return db_prescription

@app.delete("/prescriptions/{prescription_id}", status_code=204, tags=["Prescriptions"])
def delete_prescription(prescription_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    db_prescription = db.query(PrescriptionDB).filter(PrescriptionDB.id == prescription_id).first()
    if not db_prescription:
        raise HTTPException(status_code=404, detail="找不到該處方紀錄")
    db.delete(db_prescription)
    db.commit()
    return Response(status_code=204)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == '{"type":"ping"}':
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logging.error(f"WebSocket: 使用者 {user_id} 的連線發生未知錯誤: {e}")
        manager.disconnect(user_id)