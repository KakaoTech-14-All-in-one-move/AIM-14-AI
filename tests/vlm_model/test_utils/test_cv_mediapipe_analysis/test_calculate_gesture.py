# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_calculate_gesture.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.calculate_gesture import calculate_gestures_score

def test_calculate_gestures_score():
    """
    calculate_gestures_score 함수가 올바르게 계산되는지 확인합니다.
    """
    # 정상적인 점수 계산
    result = calculate_gestures_score(0.6, 0.4)
    assert result == 0.5

    # 하나의 점수가 None인 경우
    result = calculate_gestures_score(None, 0.4)
    assert result == 0.1

    # 하나의 점수가 0 이하인 경우
    result = calculate_gestures_score(-0.1, 0.4)
    assert result == 0.1

    # 두 점수가 0 이상이고 1 이하인 경우
    result = calculate_gestures_score(0.8, 0.7)
    assert result == 0.75

    # 두 점수가 모두 None인 경우
    result = calculate_gestures_score(None, None)
    assert result == 0.1
