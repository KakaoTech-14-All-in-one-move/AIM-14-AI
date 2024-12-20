# tests/vlm_model/test_utils/test_video_duration.py

import pytest
import numpy as np
import subprocess
from fastapi import HTTPException
from unittest import mock
from vlm_model.utils.video_duration import get_video_duration
from vlm_model.exceptions import VideoImportingError
import cv2

def test_get_video_duration_success(mocker):
    """
    정상적인 비디오 파일 경로를 입력으로 받아 비디오 길이가 반환되는지 확인합니다.
    """
    video_path = "/fake/path/video.mp4"

    # Mock cv2.VideoCapture
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [25.0, 2500.0]  # FPS=25, Total Frames=2500
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Call the function
    duration = get_video_duration(video_path)

    # Assertions
    cv2.VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_any_call(cv2.CAP_PROP_FPS)
    mock_cap.get.assert_any_call(cv2.CAP_PROP_FRAME_COUNT)
    mock_cap.release.assert_called_once()
    assert duration == 100.0  # 2500 / 25 = 100

def test_get_video_duration_zero_fps(mocker):
    """
    FPS가 0인 경우 기본 FPS로 설정되어 비디오 길이가 계산되는지 확인합니다.
    """
    video_path = "/fake/path/video.mp4"

    # Mock cv2.VideoCapture
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [0.0, 3000.0]  # FPS=0, Total Frames=3000
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Call the function
    duration = get_video_duration(video_path)

    # Assertions
    assert duration == 100.0  # 3000 / 30 = 100

def test_get_video_duration_no_total_frames(mocker):
    """
    총 프레임 수를 가져올 수 없는 경우 VideoImportingError가 발생하는지 확인합니다.
    """
    video_path = "/fake/path/video.mp4"

    # Mock cv2.VideoCapture
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [25.0, 0.0]  # FPS=25, Total Frames=0
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Call the function and expect an exception
    with pytest.raises(VideoImportingError) as excinfo:
        get_video_duration(video_path)

    # Assertions
    cv2.VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_any_call(cv2.CAP_PROP_FPS)
    mock_cap.get.assert_any_call(cv2.CAP_PROP_FRAME_COUNT)
    mock_cap.release.assert_not_called()
    assert "총 프레임 수를 가져올 수 없습니다." in str(excinfo.value)

def test_get_video_duration_file_not_found(mocker):
    """
    비디오 파일을 열 수 없는 경우 VideoImportingError가 발생하는지 확인합니다.
    """
    video_path = "/fake/path/nonexistent_video.mp4"

    # Mock cv2.VideoCapture
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Call the function and expect an exception
    with pytest.raises(VideoImportingError) as excinfo:
        get_video_duration(video_path)

    # Assertions
    cv2.VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_not_called()
    mock_cap.release.assert_not_called()
    assert "비디오 파일을 열 수 없습니다: /fake/path/nonexistent_video.mp4" in str(excinfo.value)

def test_get_video_duration_exception(mocker):
    """
    VideoCapture 과정에서 예외가 발생할 때 VideoImportingError가 발생하는지 확인합니다.
    """
    video_path = "/fake/path/video.mp4"

    # Mock cv2.VideoCapture to raise an exception
    mock_cap = mocker.Mock()
    mock_cap.isOpened.side_effect = Exception("Unexpected error")
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Call the function and expect an exception
    with pytest.raises(VideoImportingError) as excinfo:
        get_video_duration(video_path)

    # Assertions
    cv2.VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    assert "비디오 길이를 가져오는 중 서버 오류가 발생했습니다." in str(excinfo.value)
