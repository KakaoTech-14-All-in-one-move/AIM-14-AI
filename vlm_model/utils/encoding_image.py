# utils/encoding_image.py

import cv2
import base64
import numpy as np
from typing import Optional
import logging
from vlm_model.exceptions import ImageEncodingError
from PIL import Image

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

def encode_image(image: np.ndarray, max_size: tuple = (256, 256), quality: int = 70) -> Optional[str]:
    """
    이미지를 리사이즈하고 JPEG 형식으로 인코딩한 후 Base64 문자열로 반환합니다.

    Args:
        image (np.ndarray): 인코딩할 이미지의 NumPy 배열.
        max_size (tuple, optional): 리사이즈할 최대 크기 (가로, 세로). 기본값은 (720, 480).
        quality (int, optional): JPEG 인코딩 품질 (0-100). 기본값은 70.

    Returns:
        Optional[str]: Base64로 인코딩된 JPEG 이미지 문자열. 인코딩에 실패하면 None 반환.

    Raises:
        ImageEncodingError: 이미지 인코딩 과정에서 오류가 발생한 경우.
    """
    try:
        # 정확한 크기로 이미지 리사이즈 (비율 유지 안함)
        resized_image = cv2.resize(image, max_size, interpolation=cv2.INTER_AREA)
        logger.debug(f"이미지 정확한 크기 {max_size}으로 리사이즈 완료")

        # JPEG로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', resized_image, encode_param)
        if not result:
            logger.error("이미지 인코딩에 실패했습니다.", extra={
                "errorType": "ImageEncodingError",
                "error_message": "이미지 인코딩에 실패했습니다."
            })
            raise ImageEncodingError("이미지 인코딩에 실패했습니다.")

        # Base64 인코딩
        img_b64_str = base64.b64encode(encimg.tobytes()).decode('utf-8')
        return img_b64_str
    except ImageEncodingError:
        raise
    except Exception as e:
        logger.error(f"이미지 인코딩 중 예기치 않은 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise ImageEncodingError("이미지 인코딩 중 서버 오류가 발생했습니다.") from e
