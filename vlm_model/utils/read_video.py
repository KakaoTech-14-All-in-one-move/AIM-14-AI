# utils/read_video.py

from typing import List, Optional
import cv2
import numpy as np
import logging

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

def read_video_opencv(video_path: str, frame_indices: List[int]) -> Optional[List[np.ndarray]]:
    """
    OpenCV를 사용하여 비디오에서 특정 프레임들을 추출합니다.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return None

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
            print("지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다.")
            return None

        return frames

    except Exception as e:
        print(f"비디오에서 프레임 추출 중 오류 발생: {e}")
        return None
