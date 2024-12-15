# vlm_model/utils/image_process/add_bounding_box.py

from typing import List, Dict
from vlm_model.schemas.feedback import BoundingBox
from PIL import Image, ImageDraw
from .draw_bounding_box import draw_bounding_boxes
import logging

logger = logging.getLogger(__name__) 


def add_bounding_boxes_to_image(image: Image.Image, bounding_boxes: List[Dict[str, float]]) -> Image.Image:
    """
    이미지에 Bounding Box를 그립니다.
    
    Parameters:
    - image: PIL Image 객체
    - bounding_boxes: Bounding Box 좌표 리스트
    
    Returns:
    - image_with_boxes: Bounding Box가 그려진 이미지
    """
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        try:
            # 이미지의 실제 크기를 가져옵니다.
            width, height = image.size
            x_min = box['x_min'] * width
            y_min = box['y_min'] * height
            x_max = box['x_max'] * width
            y_max = box['y_max'] * height
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        except KeyError as e:
            logger.error(f"Bounding box 좌표 형식 오류: {e}", extra={
                "errorType": "BoundingBoxFormatError",
                "error_message": str(e)
            })
            continue  # 해당 Bounding Box는 건너뜁니다.
        except Exception as e:
            logger.error(f"Bounding box 그리기 중 오류 발생: {e}", extra={
                "errorType": "BoundingBoxDrawingError",
                "error_message": str(e)
            })
            continue  # 해당 Bounding Box는 건너뜁니다.
    return image
