# vlm_model/utils/image_process/draw_bounding_box.py

from typing import List, Dict
from PIL import Image, ImageDraw
import logging

logger = logging.getLogger(__name__) 

def draw_bounding_boxes(pil_img: Image.Image, bounding_boxes: List[Dict[str, float]], color: str = "red", width: int = 3) -> Image.Image:
    """
    PIL 이미지에 여러 개의 bounding box를 그려주는 함수.

    Parameters:
    - pil_img: PIL.Image 객체
    - bounding_boxes: List of dicts with keys 'x_min', 'y_min', 'x_max', 'y_max' (0-1 정규화된 값)
    - color: Bounding box 색상
    - width: Bounding box 선 두께

    Returns:
    - pil_img_with_boxes: Bounding box가 그려진 PIL.Image 객체
    """
    draw = ImageDraw.Draw(pil_img)
    width_px, height_px = pil_img.size

    for box in bounding_boxes:
        try:
            x_min = box['x_min']
            y_min = box['y_min']
            x_max = box['x_max']
            y_max = box['y_max']
        except KeyError as e:
            logger.error(f"Bounding box 데이터 누락: {e}", extra={
                "errorType": "BoundingBoxDataMissing",
                "error_message": f"Bounding box 데이터 누락: {e}"
            })
            continue

        tlX = int(x_min * width_px)
        tlY = int(y_min * height_px)
        brX = int(x_max * width_px)
        brY = int(y_max * height_px)

        # Bounding box 그리기
        draw.rectangle([tlX, tlY, brX, brY], outline=color, width=width)
    
    return pil_img
