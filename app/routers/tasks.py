from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_db, get_current_user
from ..models import TaskDB, PatientDB
from ..schemas import Task, TaskCreate, User


router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get("/patient/{patient_id}", response_model=List[Task])
def list_tasks_for_patient(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 僅允許病患本人或醫師查詢
    if current_user.role == "Patient":
        patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
        if not patient_profile or patient_profile.id != patient_id:
            raise HTTPException(status_code=403, detail="權限不足")
    tasks = db.query(TaskDB).filter(TaskDB.patient_id == patient_id).all()
    return tasks


