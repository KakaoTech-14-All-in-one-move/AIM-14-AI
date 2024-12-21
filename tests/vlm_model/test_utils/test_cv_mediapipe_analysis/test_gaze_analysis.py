# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_gaze_analysis.py

import pytest
from vlm_model.utils.cv_mediapipe_analysis.gaze_analysis import calculate_lack_of_eye_contact_score
from unittest.mock import MagicMock
import mediapipe as mp

def test_calculate_lack_of_eye_contact_score_center():
    """
    얼굴 랜드마크가 화면 중앙에 위치할 때 시선 부족 점수가 낮게 계산되는지 확인합니다.
    """
    # Mock face_landmarks
    mock_face_landmarks = MagicMock()
    mock_face_landmarks.landmark = {
        33: MagicMock(x=0.5, y=0.5),  # 왼쪽 눈
        263: MagicMock(x=0.5, y=0.5)  # 오른쪽 눈
    }

    frame_width = 640

    # Call the function
    result = calculate_lack_of_eye_contact_score(mock_face_landmarks, frame_width)

    # Expected score
    left_eye_x = 0.5 * 640  # 320
    right_eye_x = 0.5 * 640  # 320
    center_min = 0.4 * 640  # 256
    center_max = 0.6 * 640  # 384

    left_deviation = max(256 - 320, 320 - 384, 0)  # 0
    right_deviation = max(256 - 320, 320 - 384, 0)  # 0
    gaze_score = (left_deviation + right_deviation) / 640  # 0.0

    expected_score = round(min(gaze_score / 0.1, 1.0), 2)  # 0.0

    assert result == expected_score

def test_calculate_lack_of_eye_contact_score_off_center():
    """
    얼굴 랜드마크가 화면 중앙에서 벗어났을 때 시선 부족 점수가 높게 계산되는지 확인합니다.
    """
    # Mock face_landmarks
    mock_face_landmarks = MagicMock()
    mock_face_landmarks.landmark = {
        33: MagicMock(x=0.3, y=0.5),  # 왼쪽 눈
        263: MagicMock(x=0.7, y=0.5)  # 오른쪽 눈
    }

    frame_width = 640

    # Call the function
    result = calculate_lack_of_eye_contact_score(mock_face_landmarks, frame_width)

    # Expected score
    left_eye_x = 0.3 * 640  # 192
    right_eye_x = 0.7 * 640  # 448
    center_min = 0.4 * 640  # 256
    center_max = 0.6 * 640  # 384

    left_deviation = max(256 - 192, 192 - 384, 0)  # max(64, 0, 0) = 64
    right_deviation = max(256 - 448, 448 - 384, 0)  # max(0, 64, 0) = 64
    gaze_score = (64 + 64) / 640  # 128 / 640 = 0.2

    expected_score = round(min(0.2 / 0.1, 1.0), 2)  # min(2.0, 1.0) = 1.0

    assert result == expected_score

def test_calculate_lack_of_eye_contact_score_partial_off_center():
    """
    한쪽 눈만 화면 중앙에서 벗어났을 때 시선 부족 점수가 일부 증가하는지 확인합니다.
    """
    # Mock face_landmarks
    mock_face_landmarks = MagicMock()
    mock_face_landmarks.landmark = {
        33: MagicMock(x=0.5, y=0.5),  # 왼쪽 눈 (중앙)
        263: MagicMock(x=0.8, y=0.5)  # 오른쪽 눈 (오프 센터)
    }

    frame_width = 640

    # Call the function
    result = calculate_lack_of_eye_contact_score(mock_face_landmarks, frame_width)

    # Expected score
    left_eye_x = 0.5 * 640  # 320
    right_eye_x = 0.8 * 640  # 512
    center_min = 0.4 * 640  # 256
    center_max = 0.6 * 640  # 384

    left_deviation = max(256 - 320, 320 - 384, 0)  # 0
    right_deviation = max(256 - 512, 512 - 384, 0)  # max(0, 128, 0) = 128
    gaze_score = (0 + 128) / 640  # 128 / 640 = 0.2

    expected_score = round(min(0.2 / 0.1, 1.0), 2)  # min(2.0, 1.0) = 1.0

    assert result == expected_score
