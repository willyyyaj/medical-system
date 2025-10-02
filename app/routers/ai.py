import asyncio
import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..ai import get_openai_client
from ..ai import gemini_model
from ..utils.markdown_utils import normalize_summary_markdown
from ..auth import get_current_user
from ..schemas import TranscriptData, User
from pydantic import BaseModel


class SoapSummaryRequest(BaseModel):
    transcript: str


router = APIRouter(tags=["AI"])


@router.post("/summarize")
async def summarize_text(transcript_data: TranscriptData, current_user: User = Depends(get_current_user)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    
    # 重新導入模型以確保最新狀態
    from ..ai import gemini_model as current_gemini_model
    
    logging.info(f"Gemini 模型狀態: {current_gemini_model is not None}")
    logging.info(f"Gemini 模型類型: {type(current_gemini_model)}")
    if current_gemini_model:
        logging.info(f"Gemini 模型名稱: {getattr(current_gemini_model, 'model_name', 'Unknown')}")
    
    if not current_gemini_model:
        raise HTTPException(status_code=500, detail="Gemini 模型未能成功載入，請檢查伺服器日誌。")

    prompt = f"""
    角色：你是一位有耐心、善於溝通的家庭醫師或衛教護理師。你的專長是將複雜的醫療資訊，用溫暖、簡單易懂的語言解釋給病患聽。

    任務：請將以下的「醫病對話逐字稿」，轉換成一份給病患本人看的「看診重點摘要」。這份摘要的目的是幫助病患回家後，能清楚回顧看診內容、了解自己的狀況並遵循醫囑。

    ⚠️ 格式要求（必須嚴格遵守）：
    
    你必須使用以下 EXACT 格式，不得有任何偏差：
    
    ## 看診重點摘要
    
    **看診原因**
    [內容]
    
    **診斷結果**
    [內容]
    
    **治療計畫**
    [內容]
    
    **注意事項**
    [內容]

    🚨 重要警告：
    1. 標題必須是「## 看診重點摘要」（包含兩個井號和空格）
    2. 小標題必須是「**看診原因**」、「**診斷結果**」、「**治療計畫**」、「**注意事項**」（包含兩個星號）
    3. 每個部分之間必須有空行分隔
    4. 絕對不能省略 Markdown 格式符號（## 和 **）
    5. 如果你不按照這個格式，就是錯誤的！

    內容指引：
    - 用1-2個段落描述每個部分
    - 用白話解釋，避免專業術語
    - 嚴格基於逐字稿，不添加額外資訊
    - 重點突出最重要的診斷、治療和注意事項

    醫病對話逐字稿：
    ---
    {transcript_data.text}
    ---

    請嚴格按照上述格式生成摘要，開始：
    """

    try:
        # 添加重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await current_gemini_model.generate_content_async(prompt)
                summary_text = normalize_summary_markdown(response.text)
                logging.info(f"Gemini summary generated for user {current_user.username}")
                return {"summary": summary_text.strip()}
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # 遞增等待時間
                        logging.warning(f"API 配額限制，等待 {wait_time} 秒後重試 (嘗試 {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logging.error(f"API 配額已用完，無法生成摘要: {e}")
                        raise HTTPException(status_code=429, detail="AI 服務配額已用完，請稍後再試")
                else:
                    raise e
        raise HTTPException(status_code=500, detail=f"生成摘要失敗: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Summarization failed with Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"生成摘要失敗: {e}")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
        shutil.copyfileobj(file.file, temp_audio_file)
        temp_audio_path = temp_audio_file.name

    try:
        client = get_openai_client()
        with open(temp_audio_path, "rb") as audio_file_to_transcribe:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_to_transcribe
            )
        return {"transcript": transcript.text}
    except Exception as e:
        logging.error(f"Whisper API 轉錄失敗: {e}")
        raise HTTPException(status_code=500, detail=f"語音轉文字失敗: {e}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@router.post("/soap-summary")
async def generate_soap_summary(request: SoapSummaryRequest, current_user: User = Depends(get_current_user)):
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足")
    
    # 重新導入模型以確保最新狀態
    from ..ai import gemini_model as current_gemini_model
    
    if not current_gemini_model:
        raise HTTPException(status_code=500, detail="Gemini 模型未能成功載入，請檢查伺服器日誌。")

    prompt = f"""
    角色：你是一位專業的醫療記錄專家，專門將醫病對話逐字稿轉換成標準的 SOAP 格式醫療記錄。

    任務：請將以下的「醫病對話逐字稿」，轉換成標準的 SOAP 格式醫療記錄。

    SOAP 格式說明：
    - S (Subjective): 主觀症狀 - 病患描述的主訴、症狀、感受
    - O (Objective): 客觀發現 - 醫師觀察到的客觀事實、檢查結果、生命徵象
    - A (Assessment): 評估 - 醫師的診斷、判斷、分析
    - P (Plan): 計畫 - 治療計畫、用藥、追蹤、衛教

    重要規則：
    1. 嚴格按照 SOAP 格式分類資訊
    2. 使用專業但簡潔的醫療術語
    3. 內容必須基於逐字稿，不可添加額外資訊
    4. 使用繁體中文
    5. 每個部分都要有具體內容，如果某部分沒有資訊則標註「無」

    醫病對話逐字稿：
    ---
    {request.transcript}
    ---

    請按照以下 JSON 格式回傳 SOAP 摘要：
    {{
        "subjective": "主觀症狀內容",
        "objective": "客觀發現內容", 
        "assessment": "評估內容",
        "plan": "計畫內容"
    }}
    """

    try:
        # 添加重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await current_gemini_model.generate_content_async(prompt)
                soap_text = response.text.strip()
                break
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        logging.warning(f"SOAP API 配額限制，等待 {wait_time} 秒後重試 (嘗試 {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logging.error(f"SOAP API 配額已用完: {e}")
                        raise HTTPException(status_code=429, detail="AI 服務配額已用完，請稍後再試")
                else:
                    raise e
        
        # 嘗試解析 JSON 格式的回應
        import json
        import re
        
        # 清理回應文字，移除可能的 markdown 格式
        soap_text = re.sub(r'```json\s*', '', soap_text)
        soap_text = re.sub(r'```\s*$', '', soap_text)
        soap_text = soap_text.strip()
        
        try:
            soap_data = json.loads(soap_text)
            return soap_data
        except json.JSONDecodeError:
            # 如果無法解析 JSON，使用智能文字解析
            soap_data = {
                "subjective": "無主觀症狀描述",
                "objective": "無客觀發現", 
                "assessment": "無評估結果",
                "plan": "無治療計畫"
            }
            
            # 智能文字解析
            text_lower = soap_text.lower()
            
            # 尋找各段落
            sections = {
                'subjective': ['subjective', '主觀', 's:', '症狀'],
                'objective': ['objective', '客觀', 'o:', '發現'],
                'assessment': ['assessment', '評估', 'a:', '診斷'],
                'plan': ['plan', '計畫', 'p:', '治療']
            }
            
            for section, keywords in sections.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        # 找到關鍵字後，提取該段落內容
                        start_idx = text_lower.find(keyword)
                        if start_idx != -1:
                            # 提取從關鍵字開始到下一段落或結尾的內容
                            remaining_text = soap_text[start_idx:]
                            lines = remaining_text.split('\n')
                            content_lines = []
                            
                            for line in lines[1:]:  # 跳過包含關鍵字的第一行
                                line = line.strip()
                                if line and not any(other_keyword in line.lower() for other_section, other_keywords in sections.items() if other_section != section for other_keyword in other_keywords):
                                    content_lines.append(line)
                                elif line and any(other_keyword in line.lower() for other_section, other_keywords in sections.items() if other_section != section for other_keyword in other_keywords):
                                    break
                            
                            if content_lines:
                                soap_data[section] = '\n'.join(content_lines)
                            break
            
            return soap_data
            
    except Exception as e:
        logging.error(f"SOAP summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"生成 SOAP 摘要失敗: {e}")