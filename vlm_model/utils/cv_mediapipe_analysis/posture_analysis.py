# vlm_model/utils/cv_mediapipe_analysis/posture_analysis.py

import numpy as np
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_pose

def calculate_head_position_score(pose_landmarks, frame_width, frame_height):
    """
    머리(코 위치)의 X, Y 편차로 자세 점수를 계산합니다.

    Args:
        pose_landmarks: Mediapipe Pose 랜드마크.
        frame_width: 프레임 너비.
        frame_height: 프레임 높이.

    Returns:
        float: 자세 점수 (0: 좋음 ~ 1: 나쁨).
    """
    NOSE = mp_pose.PoseLandmark.NOSE.value
    nose_x = pose_landmarks.landmark[NOSE].x * frame_width
    nose_y = pose_landmarks.landmark[NOSE].y * frame_height

    # 화면 중심
    center_x, center_y = frame_width / 2, frame_height / 2

    # X, Y 축 편차 계산
    x_distance = abs(nose_x - center_x)
    y_distance = abs(nose_y - center_y)

    # 최대 허용 거리(정상 자세 범위)
    max_x_distance = frame_width * 0.1
    max_y_distance = frame_height * 0.2

    # X, Y 편차를 정규화하여 점수 계산
    x_score = min(x_distance / max_x_distance, 1.0)
    y_score = min(y_distance / max_y_distance, 1.0)

    # X, Y 평균
    posture_score = (x_score + y_score) / 2
    return round(posture_score, 2)  # 소수점 두 자리로 제한