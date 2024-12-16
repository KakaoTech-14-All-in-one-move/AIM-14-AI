# utils/download_video.py

import cv2
import numpy as np
from typing import Optional
import logging
from vlm_model.exceptions import VideoProcessingError
import traceback

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

def download_and_sample_video_local(video_path: str, start_time: int = 0, duration: int = 60, frame_interval: int = 1, target_size=(640, 480)) -> Optional[np.ndarray]:
    """
    주어진 비디오 파일에서 지정된 시작 시간과 지속 시간 내에서 일정 간격으로 프레임을 추출합니다.
    
    Args:
        video_path (str): 비디오 파일의 경로.
        start_time (int, optional): 추출 시작 시간(초). 기본값은 0초.
        duration (int, optional): 추출 지속 시간(초). 기본값은 60초.
        frame_interval (int, optional): 프레임 추출 간격(초). 기본값은 3초.
    
    Returns:
        Optional[np.ndarray]: 추출된 프레임의 배열 또는 실패 시 None.
    
    Raises:
        HTTPException: 파일 열기 실패 시 404, 프레임 추출 실패 시 500.
    """
    logger.info("Frame 추출을 시작합니다.")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {video_path}", extra={
                    "errorType": "VideoProcessingError",
                    "error_message": f"비디오 파일을 열 수 없습니다: {video_path}"
                })
            raise VideoProcessingError(f"비디오 파일을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logger.info(f"FPS 값을 불러올 수 없어 기본값(30.0)을 사용합니다.")
            fps = 30.0  # 기본 FPS 설정
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"비디오 FPS: {fps}, 총 프레임 수: {total_frames}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 추출할 프레임 인덱스 계산
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        frame_indices = list(range(start_frame, end_frame, int(frame_interval * fps)))
        logger.debug(f"추출할 프레임 인덱스: {frame_indices}")

        frames = []
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"프레임 {frame_counter} 읽기 실패", extra={
                    "errorType": "FrameReadError",
                    "error_message": f"프레임 {frame_counter} 읽기 실패 - 비디오가 예상보다 짧을 수 있습니다."})
                break

            if frame_counter in frame_indices:
                # 이미지 크기 조정
                frame = cv2.resize(frame, target_size)  # 지정된 크기로 리사이즈
                frames.append(frame)
                logger.debug(f"프레임 {frame_counter} 추가")
                if len(frames) == len(frame_indices):
                    break

            frame_counter += 1

        cap.release()

        if not frames:
            logger.error("지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다.", extra={
                "errorType": "VideoProcessingError",
                "error_message": "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다."
                })
            raise VideoProcessingError("지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다.")

        logger.debug(f"총 추출된 프레임 수: {len(frames)}")
        return np.array(frames)

    except VideoProcessingError as e:
        logger.error("비디오 처리 중 오류 발생", extra={
            "errorType": "VideoProcessingError",
            "error_message": e.message
        })
        raise VideoProcessingError("비디오에서 처리 중 오류가 발생했습니다.") from e
    except Exception as e:
        logger.error(f"비디오에서 프레임 추출 중 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise VideoProcessingError("비디오에서 프레임 추출 중 서버 오류가 발생했습니다.") from e