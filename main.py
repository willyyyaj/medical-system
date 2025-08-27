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
import traceback 
from sqlalchemy.orm import joinedload, selectinload
# --- Google Gemini AI 相關匯入 ---
import google.generativeai as genai

# --- JWT & 密碼處理相關匯入 ---
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# --- SQLAlchemy 相關匯入 ---
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship, backref, joinedload, selectinload
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
SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL")

# 判斷是否為 PostgreSQL URL (同時相容 'postgres://' 和 'postgresql://')
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres"):
    db_url = SQLALCHEMY_DATABASE_URL
    
    # 為了讓 SQLAlchemy 正確識別 driver，我們需要將 URL 轉換為 'postgresql+psycopg://'
    # 這個寫法可以同時處理 'postgres://' 和 'postgresql://'
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)

    logging.info("正在使用環境變數中的 PostgreSQL 資料庫。")
    engine = create_engine(db_url)
else:
    logging.warning("DATABASE_URL 環境變數未設定或格式不符，正在使用本地 SQLite。")
    SQLALCHEMY_DATABASE_URL = "sqlite:///./medical_system_final_v2.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------------------------------------------------------------
# 3. SQLAlchemy ORM 模型
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

class AppointmentDB(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    appointment_date = Column(String)
    reason = Column(String)
    summary = Column(Text, nullable=True)  
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

# --------------------------------------------------------------------------
# 4. Pydantic 模型
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
class Appointment(AppointmentBase):
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
    appointment_type: str
    created_at: datetime
    class Config:
        from_attributes = True
class SummaryUpdate(BaseModel):
    summary: str

# ✨ 修正縮排並新增 Pydantic 模型 ✨
class QuestionItem(BaseModel):
    question: str
    record_date: datetime

# --------------------------------------------------------------------------
# 5. WebSocket 連線管理器
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
# 6. FastAPI 應用與資料庫初始化
# --------------------------------------------------------------------------
app = FastAPI(title="智慧醫療資訊系統 API (V2 - 強化版)")
origins = [
    "http://localhost",
    "http://localhost:8888",
    "http://localhost:8889",
    "http://127.0.0.1:8888",
    "http://127.0.0.1:8889",
    "null",
    "https://jovial-swan-576e90.netlify.app",
    "https://fast8ambitious.netlify.app", 
    "https://medical-system-ht13.onrender.com"
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine, checkfirst=True)

# --------------------------------------------------------------------------
# 7. 依賴注入與核心認證函式
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
# 8. API 端點
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
    
    prompt = f"""
    角色：你是一位有耐心、善於溝通的家庭醫師或衛教護理師。你的專長是將複雜的醫療資訊，用溫暖、簡單易懂的語言解釋給病患聽。

任務：請將以下的「醫病對話逐字稿」，轉換成一份給病患本人看的「看診重點摘要」。這份摘要的目的是幫助病患回家後，能清楚回顧看診內容、了解自己的狀況並遵循醫囑。

摘要格式與內容指引：
- **標題**: 今日看診重點回顧
- **第一部分：您今天提到的主要問題**
  - 用1-2句話，溫和地總結病患今天來看醫生的主要原因(主訴)。例如：「您今天來看診，主要是因為最近一個禮拜感覺到喉嚨痛和咳嗽，影響到睡眠...」
- **第二部分：我的觀察與診斷**
  - 用最白話的方式，解釋醫生初步的診斷結果。避免使用艱澀術語。
  - 如果提到術語，要用比喻或簡單的描述加以說明。例如：「根據您的描述和喉嚨的狀況，這看起來是典型的上呼吸道感染，也就是我們常說的『感冒』。」
- **第三部分：接下來的治療計畫與建議**
  - 條列式說明治療計畫，讓病患一目了然。
  - **藥物方面**: 清楚說明開了哪些藥、用途是什麼、怎麼吃。例如：「我開了三天的藥，裡面包含了幫助您緩解喉嚨痛和化痰的成分。請記得三餐飯後服用。」
  - **生活建議**: 提供具體的非藥物建議。例如：「除了吃藥，請您這幾天要多喝溫開水，盡量避免吃炸的或辣的食物。」
  - **預約回診**: 如果有需要，說明下次什麼時候該回來。
- **第四部分：請注意這些情況**
  - 簡單說明需要警覺並提前回診或就醫的「警示症狀」。例如：「如果咳嗽加劇，開始發高燒超過38.5度，或感覺呼吸困難，請您要馬上回來看醫生喔。」

重要規則：
1.  **語氣溫暖，充滿同理心**：使用「您」、「我們」等詞語，讓病患感覺被關心。
2.  **絕對白話**：想像這份摘要是說給一位沒有任何醫療背景的長輩聽的。
3.  **聚焦關鍵資訊**：只摘錄對病患最重要的診斷、藥物、生活建議和回診時機。忽略不必要的對話細節。
4.  **內容準確**：摘要內容必須嚴格基於逐字稿，不可添加額外醫療建議。
5.  **使用繁體中文**。

醫病對話逐字稿：
---
{transcript_data.text}
---

請依上述格式與規則，生成給病患的「看診重點摘要」：
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
    next_appointment = db.query(AppointmentDB).filter(AppointmentDB.patient_id == patient_profile.id, AppointmentDB.appointment_date >= datetime.now().strftime("%Y-%m-%d")).order_by(AppointmentDB.appointment_date.asc()).first()
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

    appointments = db.query(AppointmentDB).options(
        joinedload(AppointmentDB.tasks),
        joinedload(AppointmentDB.doctor)
    ).filter(AppointmentDB.patient_id == patient_profile.id).all()
    
    return appointments

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
## 唯一的修改點是在 options(...) 中多加了一行 joinedload

@app.get("/doctors/me/appointments", response_model=List[AppointmentForDoctor], tags=["Doctors"])
def read_doctor_appointments(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # --- 開始執行原有邏輯 ---
        if current_user.role != "Doctor":
            raise HTTPException(status_code=403, detail="權限不足，僅限醫生本人操作")
        
        doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
        if not doctor_profile:
            raise HTTPException(status_code=404, detail="找不到對應的醫生資料")

        logging.info("--- 步驟 1: 成功獲取醫生資料 ---")

        appointments = db.query(AppointmentDB).options(
            joinedload(AppointmentDB.patient),
            selectinload(AppointmentDB.tasks)
        ).filter(AppointmentDB.doctor_id == doctor_profile.id).order_by(AppointmentDB.appointment_date.desc()).all()
        
        logging.info(f"--- 步驟 2: 成功查詢到 {len(appointments)} 筆約診紀錄 ---")
        
        # FastAPI 會在這裡嘗試將 appointments 轉換為 JSON 回應，錯誤可能發生在這之後
        # 我們先假設查詢是成功的，直接回傳
        return appointments

    except Exception as e:
        # --- 如果上方任何地方出錯，就會執行這裡 ---
        # 這個區塊會將最詳細的錯誤訊息印在您的 Render Log 中
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.error("!!!!!!!!!! /doctors/me/appointments API 發生嚴重錯誤 !!!!!!!!!!")
        logging.error(f"錯誤類型 (Error Type): {type(e).__name__}")
        logging.error(f"錯誤訊息 (Error Message): {str(e)}")
        logging.error(f"詳細追蹤 (Traceback): \n{traceback.format_exc()}")
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # 即使出錯，也回傳一個標準的 500 錯誤給前端
        raise HTTPException(status_code=500, detail="後端伺服器在處理醫師約診資料時發生未預期的錯誤。")

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

@app.post("/tasks/", response_model=Task, tags=["Tasks"], summary="建立一個通用任務")
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

@app.get("/tasks/", response_model=List[Task], tags=["Tasks"])
def get_all_my_tasks(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="權限不足")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    return patient_profile.tasks

@app.post("/appointments/{appointment_id}/tasks", response_model=Task, tags=["Tasks"], summary="為特定看診建立準備任務")
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

@app.delete("/tasks/{task_id}", status_code=204, tags=["Tasks"])
def delete_task(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="只有病患可以刪除任務")
    
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    
    db_task = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not db_task:
        return Response(status_code=204)

    if db_task.patient_id != patient_profile.id:
        raise HTTPException(status_code=403, detail="權限不足，無法刪除他人任務")
    
    db.delete(db_task)
    db.commit()
    return Response(status_code=204)

@app.post("/appointments/", response_model=Appointment, tags=["Appointments"], summary="預約未來看診")
def create_appointment(appointment: AppointmentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")

    patient_exists = db.query(PatientDB).filter(PatientDB.id == appointment.patient_id).first()
    if not patient_exists:
        raise HTTPException(status_code=404, detail="找不到指定的病患資料")
            
    db_appointment = AppointmentDB(
        **appointment.dict(), 
        doctor_id=doctor_profile.id,
        appointment_type="scheduled"
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@app.post("/appointments/walk-in", response_model=Appointment, tags=["Appointments"], summary="建立當日看診紀錄 (現場掛號)")
def create_walk_in_appointment(walk_in_data: WalkInAppointmentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")

    patient_exists = db.query(PatientDB).filter(PatientDB.id == walk_in_data.patient_id).first()
    if not patient_exists:
        raise HTTPException(status_code=404, detail="找不到指定的病患資料")
            
    appointment_time_utc = datetime.utcnow()
    
    db_appointment = AppointmentDB(
        patient_id=walk_in_data.patient_id,
        reason=walk_in_data.reason,
        appointment_date=appointment_time_utc.isoformat() + "Z",
        doctor_id=doctor_profile.id,
        created_at=appointment_time_utc,
        appointment_type="walk-in"
    )
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
                "appointment_id": appointment.id,
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

# ✨ 最終方案：使用手動建立的藥物資料庫 ✨
SIMULATED_MEDICATION_DB = {
    "A048123100": {
        "name": "PANADOL 500MG (ACETAMINOPHEN)",
        "image_url": "data:image/jpeg;base64,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", 
        "side_effects": "身體部位症狀 皮膚發疹、脫屑、發癢、發紅 消化器官噁心、嘔吐、食慾不振 神經系統頭暈、耳鳴 其他口腔潰瘍、未預期創傷或出血。"
    },
    # ... 其他藥物 ...
}

@app.get("/medication-info/{med_code}", tags=["Medications"])
def get_medication_info(med_code: str):
    if not med_code or not med_code.strip():
         raise HTTPException(status_code=400, detail="未提供有效的藥品代碼")

    info = SIMULATED_MEDICATION_DB.get(med_code)
    
    if not info:
        return {
            "name": "未知藥品",
            "image_url": "https://via.placeholder.com/100x100.png?text=No+Image",
            "side_effects": "查無此藥品的副作用資訊。"
        }
    return info

# 在 main.py 中，找到 Tasks 相關的 API 區塊

# ✨ 新增 API 端點：讓醫生根據病患ID，彙整該病患所有想問的問題 ✨
@app.get("/patients/{patient_id}/tasks/questions", response_model=List[QuestionItem], tags=["Tasks"])
def get_patient_questions_for_doctor(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    all_questions = []
    # 查詢該病患所有的 tasks
    all_tasks = db.query(TaskDB).filter(TaskDB.patient_id == patient_id).order_by(TaskDB.created_at.desc()).all()

    for task in all_tasks:
        description = task.description
        parts = description.split('|')
        question_part = next((part for part in parts if '[提問]:' in part), None)
        
        if question_part:
            question_text = question_part.replace('[提問]:', '').strip()
            if question_text and question_text != '無':
                all_questions.append(QuestionItem(
                    question=question_text,
                    record_date=task.created_at
                ))
    
    return all_questions

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