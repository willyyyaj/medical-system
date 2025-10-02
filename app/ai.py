import logging
import os
from typing import Optional

from openai import OpenAI

from .config import GOOGLE_API_KEY, OPENAI_API_KEY


gemini_model = None  # type: Optional[object]
_openai_client = None


def init_ai_sdks():
    global gemini_model
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY 環境變數未設定")
        
        # 使用最新版 API（適配 0.8.5）
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 測試連接 - 使用最新的 Gemini 2.0 模型
        try:
            # 使用 gemini-2.0-flash-exp，最新且配額更高
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            gemini_model = model
            logging.info(f"Gemini 模型已設定為: gemini-2.0-flash-exp")
            logging.info("AI SDKs 設定成功。")
        except Exception as e:
            logging.warning(f"無法使用 gemini-2.0-flash-exp，嘗試備用模型: {e}")
            try:
                # 備用模型 - 正式版 2.0
                model = genai.GenerativeModel('gemini-2.0-flash')
                gemini_model = model
                logging.info(f"使用備用模型: gemini-2.0-flash")
            except Exception as e2:
                logging.error(f"所有 Gemini 2.0 模型都無法使用: {e2}")
                raise e2
        
        # 驗證模型是否正確設定
        logging.info(f"模型驗證: {gemini_model is not None}")
        return True
    except Exception as e:
        gemini_model = None
        logging.error(f"無法設定 AI SDKs: {e}")
        logging.error(f"詳細錯誤: {str(e)}")
        return False


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變數未設定")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


