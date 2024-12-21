# vlm_model/utils/cv_mediapipe_analysis/calculate_gesture.py

def calculate_gestures_score(excessive_gestures_score, hand_movement_score):
    """
    손 관련 점수들의 평균을 계산합니다.

    Args:
        excessive_gestures_score: 과도한 제스처 점수.
        hand_movement_score: 손 움직임 점수.

    Returns:
        float: gestures 점수 (0~1 사이). 입력 값이 None이거나 감지되지 않으면 기본값 0.1을 반환.
    """
    # 기본값 설정: 값이 None이면 0.1로 초기화
    excessive_gestures_score = excessive_gestures_score if excessive_gestures_score is not None else 0.1
    hand_movement_score = hand_movement_score if hand_movement_score is not None else 0.1

    # 입력 값이 유효하지 않으면 0.1 반환
    if excessive_gestures_score <= 0.0 or hand_movement_score <= 0.0:
        return 0.1

    # 평균 계산 후 반환
    return round((excessive_gestures_score + hand_movement_score) / 2, 2)
