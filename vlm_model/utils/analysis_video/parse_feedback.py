# vlm_model/utils/analysis_video/parse_feedback_text.py

import json
import logging
from fastapi import HTTPException
from vlm_model.schemas.feedback import FeedbackSections, FeedbackDetails, BoundingBox
from vlm_model.exceptions import VideoProcessingError

# 로거 설정
logger = logging.getLogger(__name__) 


def parse_feedback_text(feedback_text: str) -> FeedbackSections:
    """
    feedback_text를 FeedbackSections 형식으로 파싱합니다.
    """
    try:
        feedback_json = json.loads(feedback_text)
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
        for section_key, field_name in {
            "gaze_processing": "gaze_processing",
            "facial_expression": "facial_expression",
            "gestures": "gestures",
            "posture_body": "posture_body",
            "movement": "movement"
        }.items():
            section = feedback_json.get(section_key, {})
            improvement = section.get("improvement", "").strip()
            recommendations = section.get("recommendations", "").strip()
            bounding_box = section.get("bounding_box", None)
            if bounding_box:
                bounding_box_obj = BoundingBox(**bounding_box)
            else:
                bounding_box_obj = None
            feedback_data[field_name] = FeedbackDetails(
                improvement=improvement,
                recommendations=recommendations,
                bounding_box=bounding_box_obj
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