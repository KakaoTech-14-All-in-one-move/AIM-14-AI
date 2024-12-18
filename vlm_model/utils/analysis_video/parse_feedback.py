# vlm_model/utils/analysis/parse_feedback_text.py

import json
import logging
import re
from fastapi import HTTPException
from vlm_model.schemas.feedback import FeedbackSections, FeedbackDetails
from vlm_model.exceptions import VideoProcessingError

# 로거 설정
logger = logging.getLogger(__name__) 


def parse_feedback_text(feedback_text: str) -> FeedbackSections:
    """
    feedback_text를 FeedbackSections 형식으로 파싱합니다.
    """
    try:
        if not feedback_text:
            logger.error("비어있는 피드백 텍스트가 전달되었습니다.", extra={
                "errorType": "EmptyFeedbackText",
                "error_message": "비어있는 피드백 텍스트가 전달되었습니다."
            })
            raise VideoProcessingError("비어있는 피드백 텍스트가 전달되었습니다.")

        # 코드 블록 제거 (```json\n ... \n```)
        clean_text = re.sub(r'^```json\s*', '', feedback_text, flags=re.MULTILINE)
        clean_text = re.sub(r'```\s*$', '', clean_text, flags=re.MULTILINE)

        feedback_json = json.loads(clean_text)

        if feedback_json.get("problem") == "none":
            return FeedbackSections(
                gaze_processing=FeedbackDetails(improvement="", recommendations=""),
                facial_expression=FeedbackDetails(improvement="", recommendations=""),
                gestures=FeedbackDetails(improvement="", recommendations=""),
                posture_body=FeedbackDetails(improvement="", recommendations=""),
                movement=FeedbackDetails(improvement="", recommendations="")
            )
        
        # 섹션별로 데이터 추출
        feedback_data = {}
        sections = ["gaze_processing", "facial_expression", "gestures", "posture_body", "movement"]
        for section_key in sections:
            section = feedback_json.get(section_key, {})
            improvement = section.get("improvement", "").strip()
            recommendations = section.get("recommendations", "").strip()
            feedback_data[section_key] = FeedbackDetails(
                improvement=improvement,
                recommendations=recommendations
            )
        
        return FeedbackSections(**feedback_data)
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 오류: {str(e)}", extra={
            "errorType": "JSONDecodeError",
            "error_message": str(e)
        })
        raise HTTPException(status_code=400, detail="JSON 디코딩 오류") from e
    except Exception as e:
        logger.error(f"FeedbackSections 생성 실패: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise HTTPException(status_code=500, detail="FeedbackSections 생성 실패") from e