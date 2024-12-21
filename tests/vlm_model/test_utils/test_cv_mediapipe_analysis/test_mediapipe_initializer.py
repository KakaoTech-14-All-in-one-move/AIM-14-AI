# tests/vlm_model/test_utils/test_cv_mediapipe_analysis/test_mediapipe_initializer.py

import pytest
from unittest import mock
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import pose, face_mesh, hands
import mediapipe as mp

def test_mediapipe_initialization(mocker):
    """
    Mediapipe 솔루션 객체들이 올바르게 초기화되는지 확인합니다.
    """
    # Mock Mediapipe Pose, FaceMesh, Hands constructors
    mock_pose_constructor = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp.solutions.pose.Pose", autospec=True)
    mock_face_mesh_constructor = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp.solutions.face_mesh.FaceMesh", autospec=True)
    mock_hands_constructor = mocker.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp.solutions.hands.Hands", autospec=True)

    # Re-import the module to trigger the initialization with mocks
    with mock.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp_pose", mock_pose_constructor):
        with mock.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp_face", mock_face_mesh_constructor):
            with mock.patch("vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer.mp_hands", mock_hands_constructor):
                import vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer  # noqa: F401

                mock_pose_constructor.assert_called_once_with(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                mock_face_mesh_constructor.assert_called_once_with(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                mock_hands_constructor.assert_called_once_with(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
