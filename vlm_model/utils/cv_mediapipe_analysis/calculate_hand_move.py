# vlm_model/utils/cv_mediapipe_analysis/calculate_hand_move.py

import numpy as np
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_hands

def calculate_hand_movement_score(current_hand_landmarks, previous_hand_landmarks, threshold=0.1):
    """
    현재 프레임과 이전 프레임에서 손목 위치 변화로 움직임 점수를 계산합니다.

    Args:
        current_hand_landmarks: 현재 프레임의 Mediapipe HandLandmarks 객체.
        previous_hand_landmarks: 이전 프레임의 Mediapipe HandLandmarks 객체.
        threshold: 움직임 점수 계산 기준이 되는 임계값.

    Returns:
        float: 움직임 점수 (0에서 1 사이).
    """
    if previous_hand_landmarks is None:
        # 이전 프레임 정보가 없으면 움직임 점수를 0.1으로 반환
        return 0.1

    WRIST = mp_hands.HandLandmark.WRIST.value
    current_wrist = current_hand_landmarks.landmark[WRIST]
    previous_wrist = previous_hand_landmarks.landmark[WRIST]

    # 손목의 이동 거리 계산
    movement_distance = np.sqrt((current_wrist.x - previous_wrist.x)**2 + (current_wrist.y - previous_wrist.y)**2)
    # threshold 대비 비율로 점수 환산
    movement_score = min(movement_distance / threshold, 1.0)
    return round(movement_score, 2)  # 소수점 두 자리로 제한