# utils/download_video.py

import cv2
import numpy as np
from typing import Optional

def download_and_sample_video_local(video_path: str, start_time: int = 0, duration: int = 60, frame_interval: int = 3) -> Optional[np.ndarray]:
    """
    주어진 비디오 파일에서 지정된 시작 시간과 지속 시간 내에서 일정 간격으로 프레임을 추출합니다.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"[WARNING] FPS 값을 불러올 수 없어 기본값(30.0)을 사용합니다.")
            fps = 30.0  # 기본 FPS 설정
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] 비디오 FPS: {fps}, 총 프레임 수: {total_frames}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 추출할 프레임 인덱스 계산
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        frame_indices = list(range(start_frame, end_frame, int(frame_interval * fps)))
        print(f"[INFO] 추출할 프레임 인덱스: {frame_indices}")

        frames = []
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] 프레임 {frame_counter} 읽기 실패")
                break

            if frame_counter in frame_indices:
                frames.append(frame)
                print(f"[INFO] 프레임 {frame_counter} 추가")
                if len(frames) == len(frame_indices):
                    break

            frame_counter += 1

        cap.release()

        if not frames:
            print("[ERROR] 지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다.")
            return None

        print(f"[INFO] 총 추출된 프레임 수: {len(frames)}")
        return np.array(frames)

    except Exception as e:
        print(f"비디오에서 프레임 추출 중 오류 발생: {e}")
        return None
