import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """驗證等級"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """驗證結果"""
    level: ValidationLevel
    message: str
    category: str
    position: Optional[Tuple[int, int]] = None
    suggestion: Optional[str] = None


@dataclass
class HighlightInfo:
    """高亮資訊"""
    text: str
    start_pos: int
    end_pos: int
    category: str
    confidence: float
    importance: str


@dataclass
class AnomalyDetection:
    """異常檢測結果"""
    value: str
    normal_range: str
    severity: str
    suggestion: str
    position: Tuple[int, int]


class MedicalSummaryValidator:
    """醫療摘要驗證 AI Agent"""
    
    def __init__(self):
        self.gemini_model = None
        self.medical_patterns = {
            'vital_signs': r'(血壓|血壓值|收縮壓|舒張壓|心率|心跳|呼吸|體溫|體溫值|脈搏)',
            'lab_values': r'(血糖|血糖值|膽固醇|血紅素|白血球|紅血球|血小板|肌酸酐|尿素氮|肝功能|腎功能)',
            'medications': r'(藥物|藥品|處方|用藥|劑量|毫克|mg|公克|g|毫升|ml)',
            'symptoms': r'(症狀|徵象|不適|疼痛|發燒|頭痛|胸痛|腹痛|噁心|嘔吐|腹瀉|便秘)',
            'diagnosis': r'(診斷|診斷結果|診斷為|疑似|可能|確診|排除)',
            'treatment': r'(治療|療程|手術|開刀|住院|出院|復健|追蹤)'
        }
        
        self.critical_values = {
            'blood_pressure': {'normal': (90, 140), 'critical': (60, 180)},
            'heart_rate': {'normal': (60, 100), 'critical': (40, 150)},
            'temperature': {'normal': (36.0, 37.5), 'critical': (35.0, 40.0)},
            'blood_sugar': {'normal': (70, 140), 'critical': (50, 300)}
        }
    
    def _get_gemini_model(self):
        """獲取 Gemini 模型，使用延遲導入"""
        if self.gemini_model is None:
            from .ai import gemini_model
            self.gemini_model = gemini_model
        return self.gemini_model

    async def validate_summary(self, transcript: str, summary: str) -> Dict[str, Any]:
        """主要驗證函數"""
        try:
            # 1. 事實一致性校驗
            fact_check_results = await self._fact_consistency_check(transcript, summary)
            
            # 2. 關鍵資訊高亮與驗證
            highlight_results = await self._extract_and_highlight_key_info(summary)
            
            # 3. 潛在遺漏提醒
            missing_alerts = await self._detect_missing_information(transcript, summary)
            
            # 4. 異常數值標記
            anomaly_results = await self._detect_anomalous_values(summary)
            
            return {
                'fact_consistency': fact_check_results,
                'highlights': highlight_results,
                'missing_alerts': missing_alerts,
                'anomalies': anomaly_results,
                'overall_score': self._calculate_overall_score(fact_check_results, missing_alerts, anomaly_results)
            }
            
        except Exception as e:
            logging.error(f"摘要驗證失敗: {e}")
            return {'error': str(e)}

    async def _fact_consistency_check(self, transcript: str, summary: str) -> List[ValidationResult]:
        """事實一致性校驗"""
        prompt = f"""
        作為醫療摘要品質控制專家，請檢查以下摘要是否與原始對話逐字稿一致：

        原始對話逐字稿：
        ---
        {transcript}
        ---

        生成的摘要：
        ---
        {summary}
        ---

        請檢查以下項目：
        1. 症狀描述是否一致
        2. 數值是否準確
        3. 診斷建議是否基於原始內容
        4. 治療計畫是否合理

        請以 JSON 格式回傳結果，使用繁體中文：
        {{
            "consistency_score": 0-100,
            "issues": [
                {{
                    "type": "symptom_mismatch|value_error|diagnosis_inconsistency|treatment_unfounded",
                    "severity": "low|medium|high|critical",
                    "description": "具體問題描述，請詳細說明哪裡不一致",
                    "suggestion": "具體的改善建議，請說明如何修正"
                }}
            ]
        }}
        """

        try:
            gemini_model = self._get_gemini_model()
            if not gemini_model:
                raise ValueError("Gemini 模型未能成功載入")
            response = await gemini_model.generate_content_async(prompt)
            
            # 清理回應文字，提取 JSON 部分
            response_text = response.text.strip()
            
            # 嘗試找到 JSON 部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # 清理可能的 markdown 格式
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            
            validation_results = []
            for issue in result.get('issues', []):
                level = ValidationLevel.WARNING if issue['severity'] == 'low' else \
                        ValidationLevel.ERROR if issue['severity'] in ['medium', 'high'] else \
                        ValidationLevel.CRITICAL
                
                validation_results.append(ValidationResult(
                    level=level,
                    message=issue['description'],
                    category=issue['type'],
                    suggestion=issue['suggestion']
                ))
            
            return validation_results
            
        except Exception as e:
            logging.error(f"事實一致性校驗失敗: {e}")
            return [ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"事實一致性校驗失敗: {str(e)}",
                category="validation_error"
            )]

    async def _extract_and_highlight_key_info(self, summary: str) -> List[HighlightInfo]:
        """關鍵資訊高亮與驗證"""
        prompt = f"""
        作為醫療資訊專家，請從以下摘要中識別並標記關鍵醫療資訊：

        摘要內容：
        ---
        {summary}
        ---

        請識別以下類型的關鍵資訊：
        1. 生命徵象數值（血壓、心率、體溫、呼吸頻率等）
        2. 實驗室檢查結果（血糖、膽固醇、血紅素等）
        3. 藥物名稱和劑量
        4. 重要症狀描述
        5. 診斷結果
        6. 治療建議

        請以 JSON 格式回傳，使用繁體中文：
        {{
            "highlights": [
                {{
                    "text": "識別到的關鍵資訊",
                    "start_pos": 起始位置,
                    "end_pos": 結束位置,
                    "category": "vital_signs|lab_values|medications|symptoms|diagnosis|treatment",
                    "confidence": 0.0-1.0,
                    "importance": "low|medium|high|critical"
                }}
            ]
        }}
        """

        try:
            gemini_model = self._get_gemini_model()
            if not gemini_model:
                raise ValueError("Gemini 模型未能成功載入")
            response = await gemini_model.generate_content_async(prompt)
            
            # 清理回應文字，提取 JSON 部分
            response_text = response.text.strip()
            
            # 嘗試找到 JSON 部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # 清理可能的 markdown 格式
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            
            highlights = []
            for item in result.get('highlights', []):
                highlights.append(HighlightInfo(
                    text=item['text'],
                    start_pos=item['start_pos'],
                    end_pos=item['end_pos'],
                    category=item['category'],
                    confidence=item['confidence'],
                    importance=item['importance']
                ))
            
            return highlights
            
        except Exception as e:
            logging.error(f"關鍵資訊高亮失敗: {e}")
            return []

    async def _detect_missing_information(self, transcript: str, summary: str) -> List[ValidationResult]:
        """潛在遺漏提醒"""
        prompt = f"""
        作為醫療品質控制專家，請檢查摘要是否遺漏了重要資訊：

        原始對話逐字稿：
        ---
        {transcript}
        ---

        生成的摘要：
        ---
        {summary}
        ---

        請檢查是否遺漏以下重要資訊，並詳細說明缺漏的具體內容：
        1. 重要症狀描述（症狀的詳細描述、持續時間、嚴重程度等）
        2. 關鍵生命徵象（血壓、心率、體溫、呼吸頻率、血氧飽和度等）
        3. 藥物過敏史（過敏藥物名稱、過敏反應類型等）
        4. 既往病史（過去疾病、手術史、慢性病等）
        5. 家族病史（家族遺傳疾病、相關疾病史等）
        6. 社會史（吸菸、飲酒、職業暴露、生活習慣等）

        請以 JSON 格式回傳，使用繁體中文，並詳細說明缺漏的具體內容：
        {{
            "missing_items": [
                {{
                    "type": "symptom|vital_sign|allergy|medical_history|family_history|social_history",
                    "severity": "low|medium|high|critical",
                    "description": "詳細說明缺漏的具體資訊內容，例如：缺漏血壓數值、缺漏頭痛症狀的詳細描述等",
                    "suggestion": "具體建議如何補充這些資訊，例如：請記錄收縮壓和舒張壓數值、請詳細描述頭痛的部位和性質等"
                }}
            ]
        }}
        """

        try:
            gemini_model = self._get_gemini_model()
            if not gemini_model:
                raise ValueError("Gemini 模型未能成功載入")
            response = await gemini_model.generate_content_async(prompt)
            
            # 清理回應文字，提取 JSON 部分
            response_text = response.text.strip()
            
            # 嘗試找到 JSON 部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # 清理可能的 markdown 格式
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            
            missing_alerts = []
            for item in result.get('missing_items', []):
                level = ValidationLevel.WARNING if item['severity'] == 'low' else \
                        ValidationLevel.ERROR if item['severity'] in ['medium', 'high'] else \
                        ValidationLevel.CRITICAL
                
                missing_alerts.append(ValidationResult(
                    level=level,
                    message=f"可能遺漏: {item['description']}",
                    category=item['type'],
                    suggestion=item['suggestion']
                ))
            
            return missing_alerts
            
        except Exception as e:
            logging.error(f"遺漏資訊檢測失敗: {e}")
            return []

    async def _detect_anomalous_values(self, summary: str) -> List[AnomalyDetection]:
        """異常數值標記"""
        # 使用正則表達式提取數值
        value_patterns = {
            'blood_pressure': r'血壓[：:]?\s*(\d+)/(\d+)',
            'heart_rate': r'心率[：:]?\s*(\d+)',
            'temperature': r'體溫[：:]?\s*(\d+\.?\d*)',
            'blood_sugar': r'血糖[：:]?\s*(\d+\.?\d*)'
        }
        
        anomalies = []
        
        for vital_type, pattern in value_patterns.items():
            matches = re.finditer(pattern, summary)
            for match in matches:
                if vital_type == 'blood_pressure':
                    systolic, diastolic = int(match.group(1)), int(match.group(2))
                    if not (self.critical_values['blood_pressure']['normal'][0] <= systolic <= self.critical_values['blood_pressure']['normal'][1]):
                        anomalies.append(AnomalyDetection(
                            value=f"{systolic}/{diastolic}",
                            normal_range="90-140/60-90",
                            severity="high" if systolic > 180 or systolic < 60 else "medium",
                            suggestion="請確認血壓數值是否正確",
                            position=(match.start(), match.end())
                        ))
                elif vital_type == 'heart_rate':
                    hr = int(match.group(1))
                    if not (self.critical_values['heart_rate']['normal'][0] <= hr <= self.critical_values['heart_rate']['normal'][1]):
                        anomalies.append(AnomalyDetection(
                            value=str(hr),
                            normal_range="60-100",
                            severity="high" if hr > 150 or hr < 40 else "medium",
                            suggestion="請確認心率數值是否正確",
                            position=(match.start(), match.end())
                        ))
                elif vital_type == 'temperature':
                    temp = float(match.group(1))
                    if not (self.critical_values['temperature']['normal'][0] <= temp <= self.critical_values['temperature']['normal'][1]):
                        anomalies.append(AnomalyDetection(
                            value=str(temp),
                            normal_range="36.0-37.5°C",
                            severity="high" if temp > 40 or temp < 35 else "medium",
                            suggestion="請確認體溫數值是否正確",
                            position=(match.start(), match.end())
                        ))
                elif vital_type == 'blood_sugar':
                    bs = float(match.group(1))
                    if not (self.critical_values['blood_sugar']['normal'][0] <= bs <= self.critical_values['blood_sugar']['normal'][1]):
                        anomalies.append(AnomalyDetection(
                            value=str(bs),
                            normal_range="70-140 mg/dL",
                            severity="high" if bs > 300 or bs < 50 else "medium",
                            suggestion="請確認血糖數值是否正確",
                            position=(match.start(), match.end())
                        ))
        
        return anomalies

    def _calculate_overall_score(self, fact_check: List[ValidationResult], 
                                missing_alerts: List[ValidationResult], 
                                anomalies: List[AnomalyDetection]) -> int:
        """計算整體品質分數"""
        base_score = 100
        
        # 事實一致性扣分
        for result in fact_check:
            if result.level == ValidationLevel.CRITICAL:
                base_score -= 20
            elif result.level == ValidationLevel.ERROR:
                base_score -= 10
            elif result.level == ValidationLevel.WARNING:
                base_score -= 5
        
        # 遺漏資訊扣分
        for result in missing_alerts:
            if result.level == ValidationLevel.CRITICAL:
                base_score -= 15
            elif result.level == ValidationLevel.ERROR:
                base_score -= 8
            elif result.level == ValidationLevel.WARNING:
                base_score -= 3
        
        # 異常數值扣分
        for anomaly in anomalies:
            if anomaly.severity == "high":
                base_score -= 12
            elif anomaly.severity == "medium":
                base_score -= 6
        
        return max(0, base_score)

    async def smart_modify_summary(self, transcript: str, summary: str) -> Dict[str, Any]:
        """AI 智能修改摘要"""
        try:
            # 1. 先進行驗證分析
            validation_result = await self.validate_summary(transcript, summary)
            
            # 2. 基於驗證結果生成修改建議
            modifications = await self._generate_modifications(transcript, summary, validation_result)
            
            # 3. 生成後端保證的安全修改版本（只做句內替換，不動 Markdown 結構與換行）
            patched_summary = self._apply_inline_replacements_preserving_structure(summary, modifications)
            
            # 添加調試日誌
            logging.info(f"AI 修改結果 - 原始摘要長度: {len(summary)}")
            logging.info(f"AI 修改結果 - 修改後摘要長度: {len(patched_summary)}")
            logging.info(f"AI 修改結果 - 修改建議數量: {len(modifications)}")
            logging.info(f"原始摘要前 100 字符: {summary[:100]}...")
            logging.info(f"修改後摘要前 100 字符: {patched_summary[:100]}...")
            
            return {
                'original_summary': summary,
                'patched_summary': patched_summary,
                'modifications': modifications,
                'validation_result': validation_result
            }
            
        except Exception as e:
            logging.error(f"智能修改失敗: {e}")
            return {'error': str(e)}

    async def _generate_modifications(self, transcript: str, summary: str, validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成具體的修改建議 - 增強可解釋性"""
        modifications = []
        
        # 添加調試日誌
        logging.info(f"AI 修改建議生成 - 逐字稿長度: {len(transcript)}, 摘要長度: {len(summary)}")
        logging.info(f"摘要前 200 字符: {summary[:200]}...")
        
        # 專注於檢測幻覺和不一致之處
        prompt = f"""
        作為醫療摘要審核專家，請仔細比較逐字稿和摘要，檢測任何不一致之處。

        逐字稿：
        {transcript}

        當前摘要（請注意保持 Markdown 格式）：
        {summary}

        請積極檢查以下問題，即使是很小的差異也要報告：

        1. **用詞差異**：摘要中的用詞是否與逐字稿完全一致？例如：「幾個月」vs「過去幾個月」
        2. **細節差異**：摘要是否遺漏了逐字稿中的重要細節？
        3. **表達方式**：摘要的表達是否與逐字稿的語氣和風格一致？
        4. **事實不一致**：摘要中的資訊是否與逐字稿不符？
        5. **數值錯誤**：摘要中的數值是否與逐字稿中的數值一致？
        6. **時間錯誤**：摘要中的時間描述是否正確？

        請以 JSON 格式回傳檢測結果，使用繁體中文：
        {{
            "modifications": [
                {{
                    "type": "replace|highlight|remove",
                    "title": "錯誤標題",
                    "description": "發現的具體錯誤",
                    "original_text": "摘要中需要修改的具體文字",
                    "correct_text": "應該替換成的正確文字",
                    "reason": "為什麼這是錯誤的",
                    "severity": "critical|high|medium|low",
                    "category": "hallucination|fact_error|value_error|time_error|diagnosis_error|treatment_error"
                }}
            ]
        }}

        ⚠️ 重要警告：你絕對不能破壞 Markdown 格式！⚠️

        重要要求：
        1. **絕對嚴禁更動 Markdown 結構**：不得修改標題格式（## 看診重點摘要）、粗體標題（**看診原因**、**診斷結果**等）、換行與空行。
        2. **僅允許句內最小範圍替換**：不得把多行合併成一行，不得修改段落結構。
        3. original_text 必須是摘要中實際存在的文字，且不包含換行字元。
        4. correct_text 必須是基於逐字稿的正確內容，且不包含換行字元。
        5. 如果檢測到幻覺內容，使用 remove 類型；若為事實錯誤，使用 replace 類型。
        6. 請同時提供該 original_text 在摘要中的字元位置：start（含）與 end（不含）。
        7. **請積極檢測錯誤**：即使是很小的用詞差異也要報告，不要過於保守。
        8. **保持完整格式**：修改後的摘要必須保持原始的 Markdown 格式，包括標題、粗體、段落分隔。
        9. **格式範例**：正確的格式應該是「## 看診重點摘要」而不是「看診重點摘要」，應該是「**看診原因**」而不是「看診原因」。
        10. **如果沒有發現錯誤，請回傳空的 modifications 陣列**：{{"modifications": []}}

        🚨 特別提醒：如果你破壞了 Markdown 格式（例如把「## 看診重點摘要」改成「看診重點摘要」），這將被視為嚴重錯誤！
        """

        try:
            gemini_model = self._get_gemini_model()
            if not gemini_model:
                raise ValueError("Gemini 模型未能成功載入")
            
            response = await gemini_model.generate_content_async(prompt)
            
            # 清理回應文字，提取 JSON 部分
            response_text = response.text.strip()
            
            # 嘗試找到 JSON 部分
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # 清理可能的 markdown 格式
            json_text = re.sub(r'```json\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            
            # 處理 AI 生成的錯誤檢測結果
            for mod in result.get('modifications', []):
                modifications.append({
                    'type': mod.get('type', 'highlight'),
                    'title': mod.get('title', '錯誤檢測'),
                    'description': mod.get('description', ''),
                    'original_text': mod.get('original_text', ''),
                    'correct_text': mod.get('correct_text', ''),
                    'reason': mod.get('reason', ''),
                    'severity': mod.get('severity', 'medium'),
                    'category': mod.get('category', 'fact_error'),
                    # 可選的精確位置，如果模型有提供
                    'start': mod.get('start'),
                    'end': mod.get('end'),
                })
            
            # 如果 AI 沒有檢測到錯誤，添加基於驗證結果的錯誤檢測
            if len(modifications) == 0:
                modifications.extend(self._generate_error_detection(transcript, summary, validation_result))
            
            return modifications[:10]  # 限制最多10個建議
            
        except Exception as e:
            logging.error(f"AI 修改建議生成失敗: {e}")
            # 回退到基於規則的建議
            return self._generate_fallback_modifications(transcript, summary, validation_result)
    
    def _generate_error_detection(self, transcript: str, summary: str, validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成錯誤檢測建議 - 專注於檢測幻覺和不一致"""
        modifications = []
        
        # 基於事實一致性問題生成錯誤檢測
        for issue in validation_result.get('fact_consistency', []):
            if issue.level in ['error', 'critical']:
                modifications.append({
                    'type': 'highlight',
                    'title': '事實不一致',
                    'description': f'摘要中的內容與逐字稿不符：{issue.message}',
                    'original_text': '摘要中的錯誤內容',
                    'correct_text': '逐字稿中的正確內容',
                    'reason': '摘要內容與原始逐字稿不一致',
                    'severity': 'high' if issue.level == 'critical' else 'medium',
                    'category': 'fact_error'
                })
        
        # 基於異常數值生成錯誤檢測
        for anomaly in validation_result.get('anomalies', []):
            if anomaly.severity in ['high', 'medium']:
                modifications.append({
                    'type': 'highlight',
                    'title': '數值異常',
                    'description': f'數值 {anomaly.value} 可能異常，正常範圍：{anomaly.normal_range}',
                    'original_text': anomaly.value,
                    'correct_text': f'請確認數值是否正確（正常範圍：{anomaly.normal_range}）',
                    'reason': '數值超出正常範圍，需要確認',
                    'severity': anomaly.severity,
                    'category': 'value_error'
                })
        
        # 檢測摘要中可能存在的幻覺（逐字稿中沒有的內容）
        transcript_lower = transcript.lower()
        summary_sentences = summary.split('。')
        
        for sentence in summary_sentences:
            if sentence.strip():
                # 檢查句子中的關鍵詞是否在逐字稿中出現
                key_terms = ['診斷', '治療', '藥物', '手術', '檢查']
                for term in key_terms:
                    if term in sentence and term not in transcript_lower:
                        modifications.append({
                            'type': 'highlight',
                            'title': '可能的幻覺內容',
                            'description': f'摘要中提到「{term}」但逐字稿中未提及',
                            'original_text': sentence,
                            'correct_text': '請確認此內容是否在逐字稿中出現',
                            'reason': '摘要中的內容在逐字稿中找不到對應',
                            'severity': 'high',
                            'category': 'hallucination'
                        })
                        break
        
        return modifications

    def _apply_inline_replacements_preserving_structure(self, summary: str, modifications: List[Dict[str, Any]]) -> str:
        """在不影響 Markdown 結構與換行的前提下，套用最小替換。

        規則：
        - 僅處理 type == 'replace' 的項目。
        - 不處理包含換行字元的 original_text / correct_text。
        - 不在以 '##', '#', '**' 開頭的標題行上進行替換。
        - 優先使用 AI 提供的 start/end 位置；若無，使用首次出現位置。
        - 找不到安全位置則跳過該項目。
        """
        if not modifications:
            return summary

        # 將摘要拆成行，保留行邊界，避免合併段落
        lines = summary.splitlines(keepends=True)
        full_text = ''.join(lines)

        # 建立不可修改範圍：標題行的區段
        protected_line_indexes = set()
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('##') or stripped.startswith('#') or stripped.startswith('**'):
                protected_line_indexes.add(idx)

        def position_in_protected_line(start: int, end: int) -> bool:
            # 將全局位置映射到行索引
            pos = 0
            for i, line in enumerate(lines):
                next_pos = pos + len(line)
                if start < next_pos and end <= next_pos:
                    return i in protected_line_indexes
                pos = next_pos
            return False

        # 逐項套用替換，從後往前避免位移影響
        # 先蒐集可用的替換區段
        spans = []  # (start, end, replacement)
        for mod in modifications:
            if mod.get('type') != 'replace':
                continue
            orig = (mod.get('original_text') or '').replace('\n', '')
            corr = (mod.get('correct_text') or '').replace('\n', '')
            if not orig or ('\n' in orig) or ('\n' in corr):
                continue

            start = mod.get('start')
            end = mod.get('end')
            # 若無座標，嘗試搜尋首次匹配位置
            if start is None or end is None:
                idx = full_text.find(orig)
                if idx == -1:
                    continue
                start, end = idx, idx + len(orig)

            # 檢查是否在受保護的標題行
            if position_in_protected_line(start, end):
                continue

            spans.append((start, end, corr))

        if not spans:
            return summary

        # 依 start 反向排序，避免位置位移
        spans.sort(key=lambda x: x[0], reverse=True)

        text = full_text
        for start, end, repl in spans:
            # 基本防禦：避免跨越換行導致結構破壞
            segment = text[start:end]
            if '\n' in segment:
                continue
            text = text[:start] + repl + text[end:]

        return text


# 全域實例
medical_validator = MedicalSummaryValidator()
