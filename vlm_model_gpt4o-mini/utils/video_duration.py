# utils/video_duration.py

import cv2
from typing import Optional

def get_video_duration(video_path: str) -> Optional[float]:
    """
    비디오 길이를 초 단위로 반환합니다.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # 기본 FPS 설정
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps
        cap.release()
        return duration
    except Exception as e:
        print(f"비디오 길이 가져오기 중 오류 발생: {e}")
        return None
