from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PrescriptionDB
from app.schemas import Prescription, PrescriptionCreate
from app.auth import get_current_user
from app.schemas import User

router = APIRouter(prefix="/prescriptions", tags=["Prescriptions"])

@router.post("/", response_model=Prescription, summary="建立處方")
def create_prescription(prescription: PrescriptionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    建立新的處方記錄。
    只有醫生可以建立處方。
    """
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="只有醫生可以建立處方")
    
    db_prescription = PrescriptionDB(
        patient_id=prescription.patient_id,
        medication_name=prescription.medication_name,
        dosage=prescription.dosage,
        frequency=prescription.frequency,
        medication_code=prescription.medication_code,
        instructions=prescription.instructions
    )
    
    db.add(db_prescription)
    db.commit()
    db.refresh(db_prescription)
    
    return db_prescription