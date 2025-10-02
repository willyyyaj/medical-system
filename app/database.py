import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import DATABASE_URL


def create_engine_from_url() -> "Engine":
    engine = None
    db_url = DATABASE_URL
    if db_url and db_url.startswith("postgres"):
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)

        logging.info("偵測到 PostgreSQL DATABASE_URL，正在建立連線引擎...")
        try:
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as _:
                logging.info("PostgreSQL 資料庫連線測試成功。")
        except Exception as e:
            logging.error(f"PostgreSQL 連線失敗: {e}")
            logging.warning("將改用本地 SQLite 資料庫。")
            engine = None

    if engine is None:
        logging.info("使用本地 SQLite 資料庫...")
        db_url = "sqlite:///./medical_system_final_v2.db"
        engine = create_engine(db_url, connect_args={"check_same_thread": False})

    return engine


engine = create_engine_from_url()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


