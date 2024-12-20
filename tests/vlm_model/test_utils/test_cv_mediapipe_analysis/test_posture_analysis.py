# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_posture_analysis.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.posture_analysis import calculate_head_position_score
from unittest.mock import MagicMock
import mediapipe as mp

def test_calculate_head_position_score_center():
    """
    머리가 화면 중앙에 위치할 때 점수가 0인지 확인합니다.
    """
    # Mock PoseLandmark 객체
    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.5, y=0.5)
    }

    frame_width = 640
    frame_height = 480

    # Call the function
    result = calculate_head_position_score(mock_pose_landmarks, frame_width, frame_height)

    # Expected score
    assert result == 0.0

def test_calculate_head_position_score_off_center():
    """
    머리가 화면 중앙에서 벗어났을 때 점수가 올바르게 계산되는지 확인합니다.
    """
    # Mock PoseLandmark 객체
    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.6, y=0.7)
    }

    frame_width = 640
    frame_height = 480

    # Calculate expected score
    nose_x = 0.6 * 640  # 384
    nose_y = 0.7 * 480  # 336
    center_x = 320
    center_y = 240

    x_distance = abs(384 - 320)  # 64
    y_distance = abs(336 - 240)  # 96

    max_x_distance = 640 * 0.1  # 64
    max_y_distance = 480 * 0.2  # 96

    x_score = min(64 / 64, 1.0)  # 1.0
    y_score = min(96 / 96, 1.0)  # 1.0

    posture_score = (1.0 + 1.0) / 2  # 1.0
    posture_score = round(posture_score, 2)  # 1.0

    # Call the function
    result = calculate_head_position_score(mock_pose_landmarks, frame_width, frame_height)

    assert result == posture_score

def test_calculate_head_position_score_partial_off_center():
    """
    머리가 X축만 벗어났을 때 점수가 올바르게 계산되는지 확인합니다.
    """
    # Mock PoseLandmark 객체
    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.6, y=0.5)
    }

    frame_width = 640
    frame_height = 480

    # Calculate expected score
    nose_x = 0.6 * 640  # 384
    nose_y = 0.5 * 480  # 240
    center_x = 320
    center_y = 240

    x_distance = abs(384 - 320)  # 64
    y_distance = abs(240 - 240)  # 0

    max_x_distance = 640 * 0.1  # 64
    max_y_distance = 480 * 0.2  # 96

    x_score = min(64 / 64, 1.0)  # 1.0
    y_score = min(0 / 96, 1.0)   # 0.0

    posture_score = (1.0 + 0.0) / 2  # 0.5
    posture_score = round(posture_score, 2)  # 0.5

    # Call the function
    result = calculate_head_position_score(mock_pose_landmarks, frame_width, frame_height)

    assert result == posture_score
