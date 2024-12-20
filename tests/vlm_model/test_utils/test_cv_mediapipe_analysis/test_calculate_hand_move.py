# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_calculate_hand_move.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.calculate_hand_move import calculate_hand_movement_score
from unittest.mock import MagicMock
import numpy as np
import mediapipe as mp

def test_calculate_hand_movement_score_with_previous():
    """
    calculate_hand_movement_score 함수가 이전 랜드마크와 현재 랜드마크의 움직임을 올바르게 계산하는지 확인합니다.
    """
    # Mock HandLandmark 객체
    mock_current_hand = MagicMock()
    mock_previous_hand = MagicMock()

    # Set landmarks
    mock_current_hand.landmark = {
        mp.solutions.hands.HandLandmark.WRIST.value: MagicMock(x=0.5, y=0.5)
    }
    mock_previous_hand.landmark = {
        mp.solutions.hands.HandLandmark.WRIST.value: MagicMock(x=0.4, y=0.4)
    }

    # Call the function
    result = calculate_hand_movement_score(mock_current_hand, mock_previous_hand, threshold=0.1)

    # Calculate expected movement
    movement_distance = np.sqrt((0.5 - 0.4)**2 + (0.5 - 0.4)**2)  # sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.1414
    expected_score = min(movement_distance / 0.1, 1.0)  # 0.1414 / 0.1 = 1.414 → clamped to 1.0
    expected_score = round(expected_score, 2)  # 1.0

    assert result == expected_score

def test_calculate_hand_movement_score_no_previous():
    """
    calculate_hand_movement_score 함수가 이전 랜드마크가 없는 경우 기본 점수를 반환하는지 확인합니다.
    """
    # Mock HandLandmark 객체
    mock_current_hand = MagicMock()

    # Set landmarks
    mock_current_hand.landmark = {
        mp.solutions.hands.HandLandmark.WRIST.value: MagicMock(x=0.5, y=0.5)
    }

    # Call the function without previous landmarks
    result = calculate_hand_movement_score(mock_current_hand, None, threshold=0.1)

    assert result == 0.1
