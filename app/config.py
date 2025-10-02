import os
import logging

from dotenv import load_dotenv


load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Secrets and security
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY 環境變數未設定，無法啟動應用程式")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# External services
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database URL (may be overridden to sqlite later)
DATABASE_URL = os.environ.get("DATABASE_URL")

# CORS origins
ORIGINS = [
    "http://localhost",
    "http://localhost:5500",
    "http://localhost:8888",
    "http://localhost:8889",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "https://fastandambitious-d6fdf.web.app",
    "https://jovial-swan-576e90.netlify.app",
    "https://medical-patient-web.web.app",
]






