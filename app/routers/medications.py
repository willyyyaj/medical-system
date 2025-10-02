from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/medication-info", tags=["Medications"])


SIMULATED_MEDICATION_DB = {
    "A048123100": {
        "name": "PANADOL 500MG (ACETAMINOPHEN)",
        "side_effects": "身體部位症狀 皮膚發疹、脫屑、發癢、發紅 消化器官噁心、嘔吐、食慾不振 神經系統頭暈、耳鳴 其他口腔潰瘍、未預期創傷或出血。",
    },
}


@router.get("/{med_code}")
def get_medication_info(med_code: str):
    if not med_code or not med_code.strip():
        raise HTTPException(status_code=400, detail="未提供有效的藥品代碼")
    info = SIMULATED_MEDICATION_DB.get(med_code)
    if not info:
        return {
            "name": "未知藥品",
            "image_url": "https://via.placeholder.com/100x100.png?text=No+Image",
            "side_effects": "查無此藥品的副作用資訊。",
        }
    return info


