from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_db, get_current_user, get_password_hash
from ..models import PatientDB, UserDB, PrescriptionDB
from ..schemas import Patient, PatientCreate, User, Prescription


router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/", response_model=List[Patient])
def list_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 醫師或病患皆可讀取病患清單（醫師用於建立預約，病患用於查詢自身資訊）
    patients = db.query(PatientDB).all()
    return patients


@router.post("/", response_model=Patient)
def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    # 建立使用者帳號
    if db.query(UserDB).filter(UserDB.username == data.credentials.username).first():
        raise HTTPException(status_code=400, detail="此帳號已存在")

    user = UserDB(
        username=data.credentials.username,
        hashed_password=get_password_hash(data.credentials.password),
        role="Patient",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    patient = PatientDB(
        name=data.name,
        birthDate=data.birthDate,
        gender=data.gender,
        user_id=user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    return patient


@router.get("/{patient_id}/prescriptions", response_model=List[Prescription])
def list_patient_prescriptions(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 醫師可查看任何病患的處方；病患只能查看自己的
    if current_user.role == "Patient":
        patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
        if not patient_profile or patient_profile.id != patient_id:
            raise HTTPException(status_code=403, detail="權限不足")
    prescriptions = db.query(PrescriptionDB).filter(PrescriptionDB.patient_id == patient_id).order_by(PrescriptionDB.created_at.desc()).all()
    return prescriptions


