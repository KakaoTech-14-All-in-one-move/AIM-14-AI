# utils/read_video.py

from typing import List, Optional
import cv2
import numpy as np
import logging
from vlm_model.exceptions import VideoProcessingError

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

def read_video_opencv(video_path: str, frame_indices: List[int]) -> Optional[List[np.ndarray]]:
    """
    OpenCV를 사용하여 비디오에서 특정 프레임들을 추출합니다.

    Args:
        video_path (str): 비디오 파일의 경로.
        frame_indices (List[int]): 추출할 프레임의 인덱스 리스트.

    Returns:
        Optional[List[np.ndarray]]: 추출된 프레임들의 리스트 또는 실패 시 None.

    Raises:
        FileNotFoundError: 비디오 파일을 열 수 없을 때.
        ValueError: 지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없을 때.
        Exception: 기타 예외 발생 시.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {video_path}", extra={
                "errorType": "VideoProcessingError",
                "error_message": f"비디오 파일을 열 수 없습니다: {video_path}"
            })
            raise VideoProcessingError(f"비디오 파일을 열 수 없습니다: {video_path}")

        frames = []
        frame_counter = 0
        frame_indices_set = set(frame_indices)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter in frame_indices_set:
                frames.append(frame)
                if len(frames) == len(frame_indices):
                    break

            frame_counter += 1

        cap.release()

        if not frames:
            logger.error("지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다.", extra={
                "errorType": "VideoProcessingError",
                "error_message": "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없음."
            })
            raise VideoProcessingError("지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없음.")

        return frames

    except VideoProcessingError as e:
        logger.error("비디오 처리 중 오류 발생", extra={
            "errorType": "VideoProcessingError",
            "error_message": e.message
        })
        raise VideoProcessingError("비디오에 처리중 오류가 발생했습니다.") from e
    except Exception as e:
        logger.error(f"비디오에서 프레임 추출 중 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise VideoProcessingError("비디오에서 프레임 추출 중 서버 오류가 발생했습니다.") from e
