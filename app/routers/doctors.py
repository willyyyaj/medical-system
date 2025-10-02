from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models import DoctorDB, PatientDB, AppointmentDB
from ..schemas import Doctor, User, Patient, AppointmentForDoctor
from ..auth import get_current_user

router = APIRouter(prefix="/doctors", tags=["Doctors"])

@router.get("/me", response_model=Doctor)
async def get_current_doctor(current_user: User = Depends(get_current_user)):
    """獲取當前醫生資訊"""
    from datetime import datetime
    return Doctor(
        id=1,
        user_id=1,
        name=current_user.username,
        specialty="內科",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

@router.post("/", response_model=Doctor)
async def create_doctor(doctor: Doctor, current_user: User = Depends(get_current_user)):
    """創建醫生（用於測試）"""
    return doctor

@router.get("/me/patients", response_model=List[Patient])
async def get_my_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """獲取當前醫生的病患列表"""
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    # 獲取醫生資料
    doctor = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor:
        # 如果沒有醫生記錄，返回所有病患（用於測試）
        patients = db.query(PatientDB).all()
        return patients
    
    # 獲取該醫生的所有病患（通過預約記錄）
    appointments = db.query(AppointmentDB).filter(AppointmentDB.doctor_id == doctor.id).all()
    patient_ids = list(set([appt.patient_id for appt in appointments]))
    
    if not patient_ids:
        # 如果沒有預約，返回所有病患供選擇
        patients = db.query(PatientDB).all()
        return patients
    
    patients = db.query(PatientDB).filter(PatientDB.id.in_(patient_ids)).all()
    return patients

@router.get("/me/appointments", response_model=List[AppointmentForDoctor])
async def get_my_appointments(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """獲取當前醫生的預約列表"""
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    # 獲取醫生資料
    doctor = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor:
        # 如果沒有醫生記錄，返回所有預約（用於測試）
        appointments = db.query(AppointmentDB).all()
    else:
        # 獲取該醫生的所有預約
        appointments = db.query(AppointmentDB).filter(AppointmentDB.doctor_id == doctor.id).all()
    
    result = []
    for appt in appointments:
        patient = db.query(PatientDB).filter(PatientDB.id == appt.patient_id).first()
        if patient:
            result.append(AppointmentForDoctor(
                id=appt.id,
                appointment_date=appt.appointment_date,
                reason=appt.reason,
                patient=patient,
                appointment_type=appt.appointment_type,
                created_at=appt.created_at,
                tasks=[],
                summary=getattr(appt, 'summary', None)
            ))
    
    return result