# vlm_model/utils/image_process/add_bounding_box.py

from typing import List, Dict
from vlm_model.schemas.feedback import BoundingBox
from PIL import Image, ImageDraw, ImageFont
import os
import logging

from vlm_model.config import FONT_PATH, FONT_SIZE  # 폰트 설정 임포트

logger = logging.getLogger(__name__) 


def add_bounding_boxes_to_image(image: Image.Image, bounding_boxes: List[Dict[str, float]]) -> Image.Image:
    """
    이미지에 Bounding Box를 그립니다. 각 Bounding Box의 카테고리에 따라 색상을 다르게 하고, 캡션을 추가합니다.
    
    Parameters:
    - image: PIL Image 객체
    - bounding_boxes: Bounding Box 좌표 리스트 (카테고리 포함)
    
    Returns:
    - image_with_boxes: Bounding Box가 그려진 이미지
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(str(FONT_PATH), size=FONT_SIZE)
        logger.debug(f"폰트를 성공적으로 로드했습니다: {FONT_PATH}, 크기: {FONT_SIZE}")
    except IOError:
        # 폰트를 찾을 수 없을 경우 기본 폰트 사용
        logger.debug(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}. 기본 폰트 사용.")
        font = ImageFont.load_default()
    
    # 카테고리에 따른 색상 매핑
    category_colors = {
        'gestures': 'blue',
        'movement': 'green'
    }

    for box in bounding_boxes:
        try:
            # 카테고리 추출
            category = box.get('category', 'unknown')
            color = category_colors.get(category, 'red')  # 기본 색상은 빨간색
            
            # 이미지의 실제 크기를 가져옵니다.
            width, height = image.size
            x_min = box['x_min'] * width
            y_min = box['y_min'] * height
            x_max = box['x_max'] * width
            y_max = box['y_max'] * height
            
            # Bounding Box 그리기
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            
            # 캡션 추가 (Bounding Box 좌측 상단)
            caption = category.capitalize()
            
            # 캡션 크기 계산 using draw.textbbox
            text_bbox = draw.textbbox((0, 0), caption, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 캡션 배경 그리기
            caption_x = x_min
            caption_y = y_min - text_height if (y_min - text_height) > 0 else y_min
            draw.rectangle(
                [caption_x, caption_y, caption_x + text_width, caption_y + text_height],
                fill=color
            )
            # 캡션 텍스트 그리기
            draw.text((caption_x, caption_y), caption, fill="white", font=font)
        
        except KeyError as e:
            logger.error(f"Bounding box 데이터 누락: {e}", extra={
                "errorType": "BoundingBoxDataMissing",
                "error_message": f"Bounding box 데이터 누락: {e}"
            })
            continue
        except Exception as e:
            logger.error(f"Bounding box 그리기 중 오류 발생: {e}", extra={
                "errorType": "BoundingBoxDrawingError",
                "error_message": str(e)
            })
            continue
    return image