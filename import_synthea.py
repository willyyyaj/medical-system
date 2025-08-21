#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 00:35:03 2025

@author: huyuwei
"""

# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

# --- 匯入我們在 main.py 中定義的 SQLAlchemy 元件 ---
# 這讓我們可以在這個獨立的腳本中，操作與 FastAPI 應用相同的資料庫
# *** 重要：請確保您的 main.py 檔案中有 PatientDB 這個類別 ***
from main import SessionLocal, PatientDB

# --------------------------------------------------------------------------
# 1. 設定路徑
# --------------------------------------------------------------------------
# Synthea™ 產生 FHIR 檔案的預設輸出路徑
# Path.home() 會自動找到您的個人資料夾 (例如 /Users/huyuwei)
SYNTHEA_OUTPUT_DIR = Path.home() / "Desktop" / "synthea" / "output" / "fhir"

def import_synthea_data():
    """
    解析 Synthea™ 產生的 FHIR JSON 檔案，並將病患資料匯入 SQLite 資料庫。
    """
    print("--- 開始匯入 Synthea™ 合成資料 ---")

    if not SYNTHEA_OUTPUT_DIR.exists():
        print(f"錯誤：找不到 Synthea™ 輸出目錄於 '{SYNTHEA_OUTPUT_DIR}'")
        print("請確認您已成功執行 ./run_synthea 並且路徑設定正確。")
        return

    db = SessionLocal()
    patient_count = 0

    try:
        # 遍歷輸出目錄中的所有 JSON 檔案
        for file_path in SYNTHEA_OUTPUT_DIR.glob("*.json"):
            print(f"正在處理檔案: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if data.get("resourceType") == "Bundle" and "entry" in data:
                    for entry in data["entry"]:
                        resource = entry.get("resource", {})
                        
                        if resource.get("resourceType") == "Patient":
                            name_data = next((n for n in resource.get("name", []) if n.get("use") == "official"), None)
                            if name_data:
                                first_name = " ".join(name_data.get("given", []))
                                last_name = name_data.get("family", "")
                                full_name = f"{last_name}{first_name}"
                            else:
                                continue

                            birth_date = resource.get("birthDate")
                            
                            gender_map = {
                                "male": "男性",
                                "female": "女性",
                                "other": "其他",
                                "unknown": "未知"
                            }
                            gender = gender_map.get(resource.get("gender", "unknown").lower())

                            if not all([full_name, birth_date, gender]):
                                continue

                            db_patient = PatientDB(
                                name=full_name,
                                birthDate=birth_date,
                                gender=gender
                            )
                            db.add(db_patient)
                            patient_count += 1
        
        print("\n正在將所有資料寫入資料庫，請稍候...")
        db.commit()
        print(f"--- 匯入完成！總共新增了 {patient_count} 位病患資料。 ---")

    except Exception as e:
        print(f"\n發生錯誤：{e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_synthea_data()