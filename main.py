# -*- coding: utf-8 -*-
import os
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ORIGINS
from app.database import Base, engine
from app.ai import init_ai_sdks
from app.routers import auth as auth_router
from app.routers import ai as ai_router
from app.routers import patients as patients_router
from app.routers import doctors as doctors_router
from app.routers import tasks as tasks_router
from app.routers import appointments as appointments_router
from app.routers import prescriptions as prescriptions_router
from app.routers import medications as medications_router
from app.routers import dashboard as dashboard_router
from app.routers import validation as validation_router


print("--- 應用程式啟動，版本 v5 ---", file=sys.stderr)


app = FastAPI(title="智慧醫療資訊系統 API (V2 - 強化版)")


@app.on_event("startup")
def on_startup() -> None:
    try:
        logging.info("應用程式啟動，正在檢查並建立資料庫表格...")
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logging.info("資料庫表格檢查完畢。")
        
        # 初始化 AI SDKs
        init_ai_sdks()
        logging.info("AI SDKs 初始化完成。")
    except Exception as e:
        logging.error(f"應用程式初始化失敗: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 掛載各模組路由
app.include_router(auth_router.router)
app.include_router(ai_router.router)
app.include_router(patients_router.router)
app.include_router(doctors_router.router)
app.include_router(tasks_router.router)
app.include_router(appointments_router.router)
app.include_router(validation_router.router)
# app.include_router(prescriptions_router.router)
# app.include_router(medications_router.router)
# app.include_router(dashboard_router.router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="127.0.0.1", port=port)

