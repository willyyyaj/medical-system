from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user, get_db
from ..models import AppointmentDB, PatientDB, TaskDB
from ..schemas import DashboardData, User


router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard", response_model=DashboardData)
def get_dashboard_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="僅限病患存取")
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    next_appointment = (
        db.query(AppointmentDB)
        .filter(
            AppointmentDB.patient_id == patient_profile.id,
            AppointmentDB.appointment_date >= datetime.now().strftime("%Y-%m-%d"),
        )
        .order_by(AppointmentDB.appointment_date.asc())
        .first()
    )
    pending_tasks = db.query(TaskDB).filter(TaskDB.patient_id == patient_profile.id, TaskDB.is_completed == False).all()
    return DashboardData(next_appointment=next_appointment, pending_tasks=pending_tasks)


