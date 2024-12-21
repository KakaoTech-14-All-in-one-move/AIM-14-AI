# vlm_model/utils/cv_mediapipe_analysis/gesture_analysis.py

import numpy as np
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_hands

def calculate_excessive_gestures_score(hand_landmarks):
    """
    과도한 손동작 정도를 점수로 반환합니다. (엄지와 검지 끝 랜드마크 거리로 예측)

    Args:
        hand_landmarks: Mediapipe Hands 랜드마크.

    Returns:
        float: 과도한 제스처 점수 (0에서 1 사이).
    """
    THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP.value
    INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP.value

    # 엄지 끝, 검지 끝 랜드마크 좌표
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_tip = hand_landmarks.landmark[INDEX_TIP]

    # 두 점 사이의 거리
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    # 0.2를 기준으로 1까지 정규화
    return round(min(distance / 0.2, 1.0), 2)  # 소수점 두 자리로 제한