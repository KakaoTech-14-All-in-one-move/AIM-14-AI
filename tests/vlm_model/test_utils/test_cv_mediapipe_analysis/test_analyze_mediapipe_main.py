# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_analyze_mediapipe_main.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import analyze_frame

@pytest.fixture
def dummy_frame():
    # 480x640 컬러 이미지 (BGR)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

def test_analyze_frame_no_landmarks(mocker, dummy_frame):
    """
    pose, face, hands 모두 랜드마크 없음.
    """
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    mock_pose.process.return_value.pose_landmarks = None
    mock_face_mesh.process.return_value.multi_face_landmarks = None
    mock_hands.process.return_value.multi_hand_landmarks = None

    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(dummy_frame)
    assert feedback == {
        "posture_score": 0.0,
        "gaze_score": 0.0,
        "gestures_score": 0.0,
        "sudden_movement_score": 0.0
    }
    assert current_pose_landmarks is None
    assert current_hand_landmarks is None

def test_analyze_frame_pose_only(mocker, dummy_frame):
    """
    pose 랜드마크만 있는 경우.
    """
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    # pose 결과 Mock
    mock_pose.process.return_value.pose_landmarks = MagicMock()
    mock_face_mesh.process.return_value.multi_face_landmarks = None
    mock_hands.process.return_value.multi_hand_landmarks = None

    # 계산 함수들도 Mock
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_head_position_score", return_value=0.8)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_sudden_movement_score", return_value=0.6)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_gestures_score", return_value=0.0)

    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(dummy_frame)
    assert feedback["posture_score"] == 0.8
    assert feedback["sudden_movement_score"] == 0.6
    assert feedback["gaze_score"] == 0.0
    assert feedback["gestures_score"] == 0.0
    assert current_pose_landmarks is not None
    assert current_hand_landmarks is None

def test_analyze_frame_pose_face(mocker, dummy_frame):
    """
    pose와 face 랜드마크가 모두 있는 경우.
    """
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    mock_pose.process.return_value.pose_landmarks = MagicMock()
    mock_face_mesh.process.return_value.multi_face_landmarks = [MagicMock()]
    mock_hands.process.return_value.multi_hand_landmarks = None

    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_head_position_score", return_value=0.5)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_sudden_movement_score", return_value=0.2)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_lack_of_eye_contact_score", return_value=0.3)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_gestures_score", return_value=0.0)

    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(dummy_frame)
    assert feedback == {
        "posture_score": 0.5,
        "gaze_score": 0.3,
        "gestures_score": 0.0,
        "sudden_movement_score": 0.2
    }
    assert current_pose_landmarks is not None
    assert current_hand_landmarks is None

def test_analyze_frame_pose_face_hands(mocker, dummy_frame):
    """
    pose, face, hands 모두 있는 경우.
    """
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    mock_face_mesh = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.face_mesh")
    mock_hands = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.hands")

    mock_pose.process.return_value.pose_landmarks = MagicMock()
    mock_face_mesh.process.return_value.multi_face_landmarks = [MagicMock()]
    mock_hand_landmarks = MagicMock()
    mock_hands.process.return_value.multi_hand_landmarks = [mock_hand_landmarks, mock_hand_landmarks]  # 두 개의 hand_landmarks 가정

    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_head_position_score", return_value=0.7)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_sudden_movement_score", return_value=0.4)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_lack_of_eye_contact_score", return_value=0.25)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_excessive_gestures_score", return_value=0.6)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_hand_movement_score", return_value=0.1)
    mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.calculate_gestures_score", return_value=0.35)

    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(dummy_frame)
    assert feedback == {
        "posture_score": 0.7,
        "gaze_score": 0.25,
        "gestures_score": 0.35,
        "sudden_movement_score": 0.4
    }
    assert current_pose_landmarks is not None
    assert current_hand_landmarks is not None

def test_analyze_frame_exception(mocker, dummy_frame):
    """
    함수 내에서 예외 발생 시 기본값 반환 테스트.
    """
    mock_pose = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main.pose")
    # 예외 발생 시키기 위해 process 호출 시 Exception 발생
    mock_pose.process.side_effect = Exception("Unexpected error")

    feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(dummy_frame)
    assert feedback == {
        "posture_score": 0.0,
        "gaze_score": 0.0,
        "gestures_score": 0.0,
        "sudden_movement_score": 0.0
    }
    assert current_pose_landmarks is None
    assert current_hand_landmarks is None
