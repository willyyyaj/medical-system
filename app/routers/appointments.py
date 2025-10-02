from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session, joinedload

# from ..ai import gemini_model  # 延遲導入
from ..auth import get_current_user, get_db
from ..models import AppointmentDB, DoctorDB, PatientDB, TaskDB
from ..schemas import Appointment, AppointmentCreate, WalkInAppointmentCreate, User, AppointmentDetail, SummaryUpdate, Task, TaskCreate


router = APIRouter(prefix="/appointments", tags=["Appointments"])


@router.post("/", response_model=Appointment, summary="預約未來看診")
def create_appointment(appointment: AppointmentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")
    patient_exists = db.query(PatientDB).filter(PatientDB.id == appointment.patient_id).first()
    if not patient_exists:
        raise HTTPException(status_code=404, detail="找不到指定的病患資料")
    db_appointment = AppointmentDB(**appointment.dict(), doctor_id=doctor_profile.id, appointment_type="scheduled")
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment


@router.post("/walk-in", response_model=Appointment, summary="建立當日看診紀錄 (現場掛號)")
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
        appointment_type="walk-in",
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment


@router.delete("/{appointment_id}", status_code=204)
def delete_appointment(appointment_id: int, db: Session = Depends(get_db)):
    db_appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="找不到該看診紀錄")
    db.delete(db_appointment)
    db.commit()
    return Response(status_code=204)


@router.get("/{appointment_id}/summary", response_model=AppointmentDetail, summary="獲取單一看診的詳細摘要")
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


@router.post("/{appointment_id}/summary", status_code=200, summary="批准並發送摘要")
async def approve_and_send_summary(appointment_id: int, summary_data: SummaryUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    doctor_profile = db.query(DoctorDB).filter(DoctorDB.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="找不到對應的醫生資料")
    appointment = db.query(AppointmentDB).options(joinedload(AppointmentDB.patient)).filter(AppointmentDB.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="找不到該看診紀錄")
    if appointment.doctor_id != doctor_profile.id:
        raise HTTPException(status_code=403, detail="權限不足，無法修改非自己的看診紀錄")
    appointment.summary = summary_data.summary
    
    # 重新導入 gemini_model 以確保最新狀態
    from ..ai import gemini_model as current_gemini_model
    logging.info(f"準備生成標籤 - gemini_model: {current_gemini_model is not None}, summary: '{summary_data.summary}'")
    if current_gemini_model and summary_data.summary:
        tagging_prompt = f"""
        角色：你是一個專業的醫療衛教助理。
        任務：請仔細分析以下的「看診摘要」，從中提取出所有對病患有用的衛教關鍵字。
        關鍵字類型應包含：
        - 疾病或症狀 (例如: 高血壓, 頭晕)
        - 飲食建議 (例如: 少鹽飲食, 戒酒, 地瓜)
        - 生活作息建議 (例如: 規律運動, 充足睡眠)
        - 藥物名稱或類型 (例如: 阿斯匹靈, 降血糖藥)
        - 追蹤指標 (例如: 血糖監測, 血壓測量)
        輸出規則：
        - 每個關鍵字都是一個簡短的詞語。
        - 所有關鍵字合併成一個單一的字串。
        - 關鍵字之間用「英文逗號」分隔。
        - 不要包含 # 符號。
        - 除了逗號分隔的關鍵字字串，不要有任何其他文字或解釋。
        看診摘要：
        ---
        {summary_data.summary}
        ---
        請生成關鍵字字串：
        """
        try:
            logging.info(f"正在為約診 {appointment.id} 生成衛教標籤...")
            response = await current_gemini_model.generate_content_async(tagging_prompt)
            generated_tags = response.text.strip()
            appointment.tags = generated_tags
            logging.info(f"成功生成標籤: {generated_tags}")
        except Exception as e:
            logging.error(f"生成衛教標籤失敗: {e}")
            appointment.tags = None
    db.commit()
    db.refresh(appointment)
    return {"message": "摘要與衛教標籤已成功儲存"}


@router.post("/{appointment_id}/tasks", response_model=Task, summary="為特定看診建立任務")
def create_appointment_task(appointment_id: int, task: TaskCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Patient":
        raise HTTPException(status_code=403, detail="權限不足，僅限病患操作")
    
    # 檢查病患身份
    patient_profile = db.query(PatientDB).filter(PatientDB.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="找不到對應的病患資料")
    
    # 檢查看診記錄是否屬於該病患
    appointment = db.query(AppointmentDB).filter(
        AppointmentDB.id == appointment_id, 
        AppointmentDB.patient_id == patient_profile.id
    ).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="找不到指定的看診紀錄，或該紀錄不屬於您")
    
    # 建立任務
    db_task = TaskDB(
        description=task.description,
        due_date=task.due_date,
        appointment_id=appointment_id,
        patient_id=patient_profile.id
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


