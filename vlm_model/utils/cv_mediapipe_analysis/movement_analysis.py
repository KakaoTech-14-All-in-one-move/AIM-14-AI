# vlm_model/utils/cv_mediapipe_analysis/movement_analysis.py

import numpy as np
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_pose

def calculate_sudden_movement_score(pose_landmarks, previous_pose_landmarks, threshold=0.1):
    """
    갑작스러운 움직임 정도를 점수로 반환합니다.
    어깨와 엉덩이의 평균 위치 변화로 추정.

    Args:
        pose_landmarks: 현재 프레임의 포즈 랜드마크.
        previous_pose_landmarks: 이전 프레임의 포즈 랜드마크.
        threshold: 움직임 감지 임계값.

    Returns:
        float: 갑작스러운 움직임 점수 (0에서 1 사이).
    """
    if previous_pose_landmarks is None:
        return 0.1

    def get_center(landmarks, points):
        x = np.mean([landmarks.landmark[p].x for p in points])
        y = np.mean([landmarks.landmark[p].y for p in points])
        return np.array([x, y])

    # 상체 중심 (양 어깨, 양 엉덩이 평균 좌표)
    current_center = get_center(pose_landmarks, [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ])
    previous_center = get_center(previous_pose_landmarks, [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ])

    # 현재 상체 중심과 이전 상체 중심 사이의 이동 거리
    movement = np.linalg.norm(current_center - previous_center)
    return round(min(movement / threshold, 1.0), 2)