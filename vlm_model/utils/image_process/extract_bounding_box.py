# vlm_model/utils/image_processing/extract_bounding_boxes.py

from typing import List, Dict
from vlm_model.schemas.feedback import FeedbackSections
import logging

logger = logging.getLogger(__name__) 

def extract_bounding_boxes(feedback_sections: FeedbackSections) -> List[Dict[str, float]]:
    """
    FeedbackSections 객체에서 특정 항목의 Bounding Box 좌표와 카테고리를 추출합니다.
    
    Parameters:
    - feedback_sections: FeedbackSections 객체
    
    Returns:
    - bounding_boxes: Bounding Box 좌표 리스트 (각 항목에 카테고리 포함)
    """
    bounding_boxes = []

    try:
        # 제스처 및 손동작 (3번)과 갑작스러운 행동 및 움직임 (5번) 항목만 처리
        sections_to_include = [
            ('gestures', feedback_sections.gestures),
            ('movement', feedback_sections.movement)
        ]

        for category, section in sections_to_include:
            if section.bounding_box:
                bounding_boxes.append({
                    'x_min': section.bounding_box.x_min,
                    'y_min': section.bounding_box.y_min,
                    'x_max': section.bounding_box.x_max,
                    'y_max': section.bounding_box.y_max,
                    'category': category  # 카테고리 정보 추가
                })
    except AttributeError as e:
        logger.error(f"Bounding box 좌표 추출 중 오류 발생: {e}", extra={
            "errorType": "BoundingBoxExtractionError",
            "error_message": str(e)
        })
        raise ValueError("Bounding box 좌표를 추출하는 중 오류가 발생했습니다.") from e
    except Exception as e:
        logger.error(f"Bounding box 좌표 추출 중 예기치 않은 오류 발생: {e}", extra={
            "errorType": "BoundingBoxExtractionError",
            "error_message": str(e)
        })
        raise ValueError("Bounding box 좌표를 추출하는 중 오류가 발생했습니다.") from e

    return bounding_boxes
