# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_analyze_mediapipe_main.py

import pytest
from unittest import mock
from unittest.mock import MagicMock
import numpy as np
import cv2
from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import (
    analyze_frame,
    calculate_gestures_score,
    calculate_hand_movement_score,
    calculate_head_position_score
)

from vlm_model.exceptions import VideoImportingError
import mediapipe as mp

@pytest.fixture
def mock_pose_landmarks():
    mock_landmarks = MagicMock()
    mock_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.5, y=0.5),
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: MagicMock(x=0.4, y=0.4),
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: MagicMock(x=0.6, y=0.4)
    }
    return mock_landmarks

@pytest.fixture
def mock_face_landmarks():
    mock_landmarks = MagicMock()
    mock_landmarks.landmark = {
        33: MagicMock(x=0.5, y=0.5),  # 왼쪽 눈
        263: MagicMock(x=0.5, y=0.5)  # 오른쪽 눈
    }
    return mock_landmarks

@pytest.fixture
def mock_hand_landmarks():
    mock_landmarks = MagicMock()
    mock_landmarks.landmark = {
        mp.solutions.hands.HandLandmark.WRIST.value: MagicMock(x=0.5, y=0.5),
        mp.solutions.hands.HandLandmark.THUMB_TIP.value: MagicMock(x=0.6, y=0.6),
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value: MagicMock(x=0.7, y=0.7)
    }
    return mock_landmarks

def test_analyze_frame_success(mocker, mock_pose_landmarks, mock_face_landmarks, mock_hand_landmarks):
    """
    정상적인 프레임을 분석하여 올바른 피드백을 반환하는지 확인합니다.
    """
    # Mock Mediapipe Pose, FaceMesh, Hands
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    # Configure the process() methods to return mock results
    mock_pose.process.return_value.pose_landmarks = mock_pose_landmarks
    mock_face_mesh.process.return_value.multi_face_landmarks = [mock_face_landmarks]
    mock_hands.process.return_value.multi_hand_landmarks = [mock_hand_landmarks]

    # Mock the helper functions
    mock_calculate_head_position_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_head_position_score",
        return_value=0.75
    )
    mock_calculate_sudden_movement_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_sudden_movement_score",
        return_value=0.65
    )
    mock_calculate_lack_of_eye_contact_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_lack_of_eye_contact_score",
        return_value=0.55
    )
    mock_calculate_excessive_gestures_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_excessive_gestures_score",
        return_value=0.45
    )
    mock_calculate_hand_movement_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_hand_movement_score",
        return_value=0.35
    )
    mock_calculate_gestures_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_gestures_score",
        return_value=0.4
    )

    # Create a fake frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Call the function
    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(frame)

    # Assertions
    mock_pose.process.assert_called_once()
    mock_face_mesh.process.assert_called_once()
    mock_hands.process.assert_called_once()

    mock_calculate_head_position_score.assert_called_once_with(mock_pose_landmarks, 640, 480)
    mock_calculate_sudden_movement_score.assert_called_once_with(mock_pose_landmarks, None)
    mock_calculate_lack_of_eye_contact_score.assert_called_once_with(mock_face_landmarks, 640)
    mock_calculate_excessive_gestures_score.assert_called_once_with(mock_hand_landmarks)
    mock_calculate_hand_movement_score.assert_not_called()  # previous_hand_landmarks is None
    mock_calculate_gestures_score.assert_called_once_with(0.45, 0.35)

    assert feedback == {
        "posture_score": 0.75,
        "gaze_score": 0.55,
        "gestures_score": 0.4,
        "sudden_movement_score": 0.65
    }
    assert current_pose_landmarks == mock_pose_landmarks
    assert current_hand_landmarks == mock_hand_landmarks

def test_analyze_frame_no_landmarks(mocker):
    """
    포즈, 얼굴, 손 랜드마크가 없는 프레임을 분석할 때 기본 점수가 반환되는지 확인합니다.
    """
    # Mock Mediapipe Pose, FaceMesh, Hands
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    # Configure the process() methods to return no landmarks
    mock_pose.process.return_value.pose_landmarks = None
    mock_face_mesh.process.return_value.multi_face_landmarks = None
    mock_hands.process.return_value.multi_hand_landmarks = None

    # Mock the helper functions (should not be called)
    mock_calculate_head_position_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_head_position_score"
    )
    mock_calculate_sudden_movement_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_sudden_movement_score"
    )
    mock_calculate_lack_of_eye_contact_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_lack_of_eye_contact_score"
    )
    mock_calculate_excessive_gestures_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_excessive_gestures_score"
    )
    mock_calculate_hand_movement_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_hand_movement_score"
    )
    mock_calculate_gestures_score = mocker.patch(
        "vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_gestures_score"
    )

    # Create a fake frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Call the function
    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(frame)

    # Assertions
    mock_pose.process.assert_called_once()
    mock_face_mesh.process.assert_called_once()
    mock_hands.process.assert_called_once()

    mock_calculate_head_position_score.assert_not_called()
    mock_calculate_sudden_movement_score.assert_not_called()
    mock_calculate_lack_of_eye_contact_score.assert_not_called()
    mock_calculate_excessive_gestures_score.assert_not_called()
    mock_calculate_hand_movement_score.assert_not_called()
    mock_calculate_gestures_score.assert_called_once_with(0.0, 0.0)

    assert feedback == {
        "posture_score": 0.0,
        "gaze_score": 0.0,
        "gestures_score": 0.0,
        "sudden_movement_score": 0.0
    }
    assert current_pose_landmarks is None
    assert current_hand_landmarks is None

def test_analyze_frame_exception(mocker):
    """
    analyze_frame 함수 내에서 예외가 발생할 때 기본값이 반환되는지 확인합니다.
    """
    # Mock Mediapipe Pose, FaceMesh, Hands
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    # Configure the process() method to raise an exception
    mock_pose.process.side_effect = Exception("Unexpected error")

    # Create a fake frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Call the function
    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(frame)

    # Assertions
    mock_pose.process.assert_called_once()
    mock_face_mesh.process.assert_not_called()
    mock_hands.process.assert_not_called()

    assert feedback == {
        "posture_score": 0.0,
        "gaze_score": 0.0,
        "gestures_score": 0.0,
        "sudden_movement_score": 0.0
    }
    assert current_pose_landmarks is None
    assert current_hand_landmarks is None

def test_calculate_gestures_score():
    """
    calculate_gestures_score 함수가 올바르게 계산되는지 확인합니다.
    """
    from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import calculate_gestures_score

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

def test_calculate_hand_movement_score_with_previous():
    """
    calculate_hand_movement_score 함수가 이전 랜드마크와 현재 랜드마크의 움직임을 올바르게 계산하는지 확인합니다.
    """
    from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import calculate_hand_movement_score

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
    from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import calculate_hand_movement_score

    # Mock HandLandmark 객체
    mock_current_hand = MagicMock()

    # Set landmarks
    mock_current_hand.landmark = {
        mp.solutions.hands.HandLandmark.WRIST.value: MagicMock(x=0.5, y=0.5)
    }

    # Call the function without previous landmarks
    result = calculate_hand_movement_score(mock_current_hand, None, threshold=0.1)

    assert result == 0.1

def test_calculate_head_position_score():
    """
    calculate_head_position_score 함수가 머리 위치에 따른 점수를 올바르게 계산하는지 확인합니다.
    """
    from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import calculate_head_position_score

    # Mock PoseLandmark 객체
    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.55, y=0.6)
    }

    # Frame dimensions
    frame_width = 640
    frame_height = 480

    # Call the function
    result = calculate_head_position_score(mock_pose_landmarks, frame_width, frame_height)

    # Calculate expected score
    nose_x = 0.55 * 640  # 352
    nose_y = 0.6 * 480   # 288
    center_x = 320
    center_y = 240

    x_distance = abs(352 - 320)  # 32
    y_distance = abs(288 - 240)  # 48

    max_x_distance = 640 * 0.1  # 64
    max_y_distance = 480 * 0.2  # 96

    x_score = min(32 / 64, 1.0)  # 0.5
    y_score = min(48 / 96, 1.0)  # 0.5

    posture_score = (x_score + y_score) / 2  # 0.5
    posture_score = round(posture_score, 2)  # 0.5

    assert result == posture_score

def test_calculate_head_position_score_edge():
    """
    calculate_head_position_score 함수가 머리 위치가 최대 허용 범위를 초과할 때 클램핑되는지 확인합니다.
    """
    from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import calculate_head_position_score

    # Mock PoseLandmark 객체
    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = {
        mp.solutions.pose.PoseLandmark.NOSE.value: MagicMock(x=0.8, y=0.9)  # Far from center
    }

    # Frame dimensions
    frame_width = 640
    frame_height = 480

    # Call the function
    result = calculate_head_position_score(mock_pose_landmarks, frame_width, frame_height)

    # Calculate expected score
    nose_x = 0.8 * 640  # 512
    nose_y = 0.9 * 480  # 432
    center_x = 320
    center_y = 240

    x_distance = abs(512 - 320)  # 192
    y_distance = abs(432 - 240)  # 192

    max_x_distance = 640 * 0.1  # 64
    max_y_distance = 480 * 0.2  # 96

    x_score = min(192 / 64, 1.0)  # 1.0
    y_score = min(192 / 96, 1.0)  # 2.0 → clamped to 1.0

    posture_score = (x_score + y_score) / 2  # (1.0 + 1.0) / 2 = 1.0
    posture_score = round(posture_score, 2)  # 1.0

    assert result == posture_score
