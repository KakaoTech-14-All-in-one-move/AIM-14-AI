# utils/video_duration.py

import cv2
from typing import Optional
import logging
from vlm_model.exceptions import VideoImportingError
import traceback

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

def get_video_duration(video_path: str) -> Optional[float]:
    """
    비디오 길이를 초 단위로 반환합니다.
    
    Args:
        video_path (str): 비디오 파일의 경로.
    
    Returns:
        Optional[float]: 비디오 길이(초) 또는 실패 시 None.
    
    Raises:
        FileNotFoundError: 비디오 파일을 열 수 없을 때.
        ValueError: FPS 값을 가져올 수 없을 때.
        Exception: 기타 예외 발생 시.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {video_path}", extra={
                "errorType": "VideoImportingError",
                "error_message": f"비디오 파일을 열 수 없습니다: {video_path}"
            })
            raise VideoImportingError(f"비디오 파일을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # 기본 FPS 설정
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames <= 0:
            logger.error("총 프레임 수를 가져올 수 없습니다.", extra={
                "errorType": "VideoImportingError",
                "error_message": "총 프레임 수를 가져올 수 없습니다."
            })
            raise VideoImportingError("총 프레임 수를 가져올 수 없습니다.")

        duration = total_frames / fps
        cap.release()
        return duration

    except VideoImportingError as e:
        logger.error("비디오 가져와서 처리 중 오류 발생", extra={
            "errorType": "VideoImportingError",
            "error_message": e.message
        })
        raise VideoImportingError("비디오를 처리하는 중에 오류가 발생했습니다.") from e
    except Exception as e:
        logger.error(f"비디오 길이를 가져오는 중 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise VideoImportingError("비디오 길이를 가져오는 중 서버 오류가 발생했습니다.") from e
