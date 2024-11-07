# utils/visualization.py

import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple

def plot_problematic_frames(frames: List[Tuple[np.ndarray, int, int, str]], feedbacks: List[str]) -> None:
    if not frames:
        print("문제 있는 프레임이 없습니다.")
        return

    for i, (frame_info, feedback) in enumerate(zip(frames, feedbacks)):
        frame, segment_number, frame_number, timestamp = frame_info

        # 피드백을 콘솔에 출력
        print(f"Segment {segment_number}, Frame {frame_number} ({timestamp})")
        print("피드백:")
        print(feedback)
        print("-" * 50)

        # OpenCV의 BGR 이미지를 RGB로 변환하여 matplotlib에서 올바르게 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 이미지 표시
        plt.figure(figsize=(8, 6))
        plt.imshow(frame_rgb)
        plt.title(f"Segment {segment_number}, Frame {frame_number} ({timestamp})", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
