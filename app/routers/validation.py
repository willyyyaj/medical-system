import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from ..auth import get_current_user
from ..schemas import User
from ..ai_agent import medical_validator
from ..utils.markdown_utils import normalize_summary_markdown


router = APIRouter(prefix="/validation", tags=["AI Validation"])


class ValidationRequest(BaseModel):
    transcript: str
    summary: str


class SmartModifyRequest(BaseModel):
    transcript: str
    summary: str


class ValidationResponse(BaseModel):
    fact_consistency: list
    highlights: list
    missing_alerts: list
    anomalies: list
    overall_score: int
    recommendations: list


@router.post("/validate-summary", response_model=ValidationResponse, summary="AI 摘要品質驗證")
async def validate_medical_summary(
    request: ValidationRequest, 
    current_user: User = Depends(get_current_user)
):
    """
    AI Agent 進行醫療摘要品質驗證
    
    功能包括：
    1. 事實一致性校驗
    2. 關鍵資訊高亮與驗證
    3. 潛在遺漏提醒
    4. 異常數值標記
    """
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    try:
        logging.info(f"開始進行摘要驗證 - 用戶: {current_user.username}")
        
        # 執行 AI Agent 驗證
        validation_result = await medical_validator.validate_summary(
            transcript=request.transcript,
            summary=request.summary
        )
        
        if 'error' in validation_result:
            raise HTTPException(status_code=500, detail=f"驗證過程發生錯誤: {validation_result['error']}")
        
        # 生成改善建議
        recommendations = _generate_recommendations(validation_result)
        
        # 格式化回應
        response_data = {
            'fact_consistency': [
                {
                    'level': result.level.value,
                    'message': result.message,
                    'category': result.category,
                    'suggestion': result.suggestion
                }
                for result in validation_result['fact_consistency']
            ],
            'highlights': [
                {
                    'text': highlight.text,
                    'start_pos': highlight.start_pos,
                    'end_pos': highlight.end_pos,
                    'category': highlight.category,
                    'confidence': highlight.confidence,
                    'importance': highlight.importance
                }
                for highlight in validation_result['highlights']
            ],
            'missing_alerts': [
                {
                    'level': result.level.value,
                    'message': result.message,
                    'category': result.category,
                    'suggestion': result.suggestion
                }
                for result in validation_result['missing_alerts']
            ],
            'anomalies': [
                {
                    'value': anomaly.value,
                    'normal_range': anomaly.normal_range,
                    'severity': anomaly.severity,
                    'suggestion': anomaly.suggestion,
                    'position': anomaly.position
                }
                for anomaly in validation_result['anomalies']
            ],
            'overall_score': validation_result['overall_score'],
            'recommendations': recommendations
        }
        
        logging.info(f"摘要驗證完成 - 整體分數: {validation_result['overall_score']}")
        return ValidationResponse(**response_data)
        
    except Exception as e:
        logging.error(f"摘要驗證失敗: {e}")
        raise HTTPException(status_code=500, detail=f"摘要驗證失敗: {str(e)}")


@router.post("/smart-modify", summary="AI 智能修改摘要")
async def smart_modify_summary(
    request: SmartModifyRequest, 
    current_user: User = Depends(get_current_user)
):
    """
    AI 智能修改醫療摘要
    
    功能包括：
    1. 主動修改摘要內容
    2. 標記修改位置和類型
    3. 提供修改說明
    """
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    try:
        logging.info(f"開始進行 AI 智能修改 - 用戶: {current_user.username}")
        
        # 正規化摘要 Markdown，確保格式正確（標題、粗體小節、空行）
        normalized_summary = normalize_summary_markdown(request.summary)

        # 執行 AI 智能修改
        modification_result = await medical_validator.smart_modify_summary(
            transcript=request.transcript,
            summary=normalized_summary
        )
        
        if 'error' in modification_result:
            raise HTTPException(status_code=500, detail=f"智能修改過程發生錯誤: {modification_result['error']}")
        
        logging.info(f"AI 智能修改完成 - 修改項目: {len(modification_result.get('modifications', []))}")
        # 再次保險：對 patched_summary 做正規化（若存在）
        if isinstance(modification_result, dict):
            patched = modification_result.get('patched_summary')
            if isinstance(patched, str) and patched:
                modification_result['patched_summary'] = normalize_summary_markdown(patched)
        return modification_result
        
    except Exception as e:
        logging.error(f"AI 智能修改失敗: {e}")
        raise HTTPException(status_code=500, detail=f"AI 智能修改失敗: {str(e)}")


def _generate_recommendations(validation_result: Dict[str, Any]) -> list:
    """生成改善建議"""
    recommendations = []
    
    # 基於事實一致性問題的建議
    fact_issues = validation_result.get('fact_consistency', [])
    if fact_issues:
        recommendations.append({
            'type': 'fact_consistency',
            'priority': 'high',
            'title': '事實一致性問題',
            'description': f'發現 {len(fact_issues)} 個事實一致性問題，建議重新檢查摘要內容',
            'actions': ['檢查症狀描述是否準確', '確認數值是否正確', '驗證診斷建議的合理性']
        })
    
    # 基於遺漏資訊的建議
    missing_alerts = validation_result.get('missing_alerts', [])
    if missing_alerts:
        recommendations.append({
            'type': 'missing_information',
            'priority': 'medium',
            'title': '資訊遺漏提醒',
            'description': f'可能遺漏 {len(missing_alerts)} 項重要資訊',
            'actions': ['檢查是否包含所有重要症狀', '確認生命徵象完整性', '補充必要的病史資訊']
        })
    
    # 基於異常數值的建議
    anomalies = validation_result.get('anomalies', [])
    if anomalies:
        recommendations.append({
            'type': 'anomalous_values',
            'priority': 'high',
            'title': '異常數值檢測',
            'description': f'發現 {len(anomalies)} 個異常數值',
            'actions': ['重新確認數值準確性', '檢查測量單位', '考慮是否需要重新測量']
        })
    
    # 基於整體分數的建議
    overall_score = validation_result.get('overall_score', 0)
    if overall_score < 70:
        recommendations.append({
            'type': 'overall_quality',
            'priority': 'critical',
            'title': '摘要品質需要改善',
            'description': f'整體品質分數為 {overall_score}，建議大幅修改摘要內容',
            'actions': ['重新生成摘要', '手動檢查所有內容', '尋求同事協助審核']
        })
    elif overall_score < 85:
        recommendations.append({
            'type': 'overall_quality',
            'priority': 'medium',
            'title': '摘要品質可進一步提升',
            'description': f'整體品質分數為 {overall_score}，建議進行小幅調整',
            'actions': ['檢查標記的問題', '補充遺漏資訊', '確認異常數值']
        })
    
    return recommendations


@router.get("/validation-stats", summary="獲取驗證統計資訊")
async def get_validation_stats(current_user: User = Depends(get_current_user)):
    """獲取 AI Agent 驗證統計資訊"""
    if current_user.role != "Doctor":
        raise HTTPException(status_code=403, detail="權限不足，僅限醫生操作")
    
    return {
        "ai_agent_status": "active",
        "supported_validations": [
            "事實一致性校驗",
            "關鍵資訊高亮",
            "潛在遺漏提醒", 
            "異常數值標記"
        ],
        "validation_categories": [
            "symptom_mismatch",
            "value_error", 
            "diagnosis_inconsistency",
            "treatment_unfounded",
            "vital_signs",
            "lab_values",
            "medications",
            "symptoms",
            "diagnosis",
            "treatment"
        ]
    }
