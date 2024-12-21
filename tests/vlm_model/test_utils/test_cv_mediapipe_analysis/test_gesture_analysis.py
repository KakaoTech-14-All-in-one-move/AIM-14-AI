# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_gesture_analysis.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.gesture_analysis import calculate_excessive_gestures_score
from unittest.mock import MagicMock
import mediapipe as mp

def test_calculate_excessive_gestures_score_within_threshold():
    """
    엄지와 검지 끝 사이의 거리가 임계값 이내일 때 점수가 올바르게 계산되는지 확인합니다.
    """
    # Mock hand_landmarks
    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = {
        mp.solutions.hands.HandLandmark.THUMB_TIP.value: MagicMock(x=0.5, y=0.5),
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value: MagicMock(x=0.55, y=0.55)
    }

    # Calculate expected distance
    distance = np.sqrt((0.5 - 0.55)**2 + (0.5 - 0.55)**2)  # sqrt(0.0025 + 0.0025) = sqrt(0.005) ≈ 0.0707
    expected_score = round(min(distance / 0.2, 1.0), 2)  # 0.0707 / 0.2 ≈ 0.35 → 0.35

    # Call the function
    result = calculate_excessive_gestures_score(mock_hand_landmarks)

    assert result == expected_score

def test_calculate_excessive_gestures_score_above_threshold():
    """
    엄지와 검지 끝 사이의 거리가 임계값을 초과할 때 점수가 클램핑되는지 확인합니다.
    """
    # Mock hand_landmarks
    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = {
        mp.solutions.hands.HandLandmark.THUMB_TIP.value: MagicMock(x=0.2, y=0.2),
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value: MagicMock(x=0.6, y=0.6)
    }

    # Calculate expected distance
    distance = np.sqrt((0.2 - 0.6)**2 + (0.2 - 0.6)**2)  # sqrt(0.16 + 0.16) = sqrt(0.32) ≈ 0.5657
    expected_score = round(min(distance / 0.2, 1.0), 2)  # 0.5657 / 0.2 ≈ 2.828 → clamped to 1.0

    # Call the function
    result = calculate_excessive_gestures_score(mock_hand_landmarks)

    assert result == 1.0

def test_calculate_excessive_gestures_score_zero_distance():
    """
    엄지와 검지 끝 사이의 거리가 0일 때 점수가 올바르게 계산되는지 확인합니다.
    """
    # Mock hand_landmarks
    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = {
        mp.solutions.hands.HandLandmark.THUMB_TIP.value: MagicMock(x=0.5, y=0.5),
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value: MagicMock(x=0.5, y=0.5)
    }

    # Call the function
    result = calculate_excessive_gestures_score(mock_hand_landmarks)

    assert result == 0.0

def test_calculate_excessive_gestures_score_none_landmarks(mocker):
    """
    손 랜드마크가 None일 때 함수가 예외를 발생시키는지 확인합니다.
    """
    # Mock hand_landmarks
    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = {}

    # Call the function and expect an AttributeError
    with pytest.raises(AttributeError):
        calculate_excessive_gestures_score(mock_hand_landmarks)
