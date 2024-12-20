# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_movement_analysis.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.movement_analysis import calculate_sudden_movement_score
from unittest.mock import MagicMock
import numpy as np
import mediapipe as mp

def test_calculate_sudden_movement_score_with_previous():
    """
    calculate_sudden_movement_score 함수가 이전 랜드마크와 현재 랜드마크의 움직임을 올바르게 계산하는지 확인합니다.
    """
    # Mock PoseLandmark 객체
    mock_current_pose = MagicMock()
    mock_previous_pose = MagicMock()

    # Set landmarks
    mock_current_pose.landmark = {
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: MagicMock(x=0.4, y=0.4),
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: MagicMock(x=0.6, y=0.4)
    }
    mock_previous_pose.landmark = {
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: MagicMock(x=0.35, y=0.35),
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: MagicMock(x=0.65, y=0.35)
    }

    # Calculate expected movement
    current_center = np.array([0.4, 0.4, 0.6, 0.4]).reshape(2, 2).mean(axis=0)  # [0.5, 0.4]
    previous_center = np.array([0.35, 0.35, 0.65, 0.35]).reshape(2, 2).mean(axis=0)  # [0.5, 0.35]
    movement_distance = np.linalg.norm(current_center - previous_center)  # sqrt(0 + 0.05^2) = 0.05
    expected_score = round(min(0.05 / 0.1, 1.0), 2)  # 0.05 / 0.1 = 0.5

    # Call the function
    result = calculate_sudden_movement_score(mock_current_pose, mock_previous_pose, threshold=0.1)

    assert result == expected_score

def test_calculate_sudden_movement_score_no_previous():
    """
    calculate_sudden_movement_score 함수가 이전 랜드마크가 없는 경우 기본 점수를 반환하는지 확인합니다.
    """
    # Mock PoseLandmark 객체
    mock_current_pose = MagicMock()

    # Call the function without previous landmarks
    result = calculate_sudden_movement_score(mock_current_pose, None, threshold=0.1)

    assert result == 0.1
