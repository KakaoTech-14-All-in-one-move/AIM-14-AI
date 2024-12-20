# tests/vlm_model/test_utils/test_download_video.py

import pytest
import numpy as np
from unittest import mock
from vlm_model.utils.download_video import download_and_sample_video_local
from vlm_model.exceptions import VideoProcessingError
import cv2

def test_download_and_sample_video_local_success(mocker):
    video_path = "/fake/video.mp4"
    start_time = 0
    duration = 60
    frame_interval = 1
    target_size = (256, 256)

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 1800]  # FPS, 총 프레임 수
    # Mock read() 메서드: 정상적인 프레임 반환
    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(1800)]
    mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("cv2.resize", side_effect=lambda frame, size, interpolation: np.resize(frame, size + (3,)))

    # 함수 호출
    result = download_and_sample_video_local(video_path, start_time, duration, frame_interval, target_size)

    # 프레임 수 확인
    expected_num_frames = duration // frame_interval
    assert result.shape == (expected_num_frames, *target_size, 3)

def test_download_and_sample_video_local_video_not_found(mocker):
    video_path = "/fake/nonexistent_video.mp4"

    # cv2.VideoCapture를 모킹하여 파일을 열 수 없다고 설정
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(video_path)

    assert f"비디오 파일을 열 수 없습니다: {video_path}" in str(excinfo.value)

def test_download_and_sample_video_local_no_frames_found(mocker):
    video_path = "/fake/video.mp4"
    start_time = 0
    duration = 60
    frame_interval = 1
    target_size = (256, 256)

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 1800]  # FPS, 총 프레임 수
    # Mock read() 메서드: 프레임을 반환하지 않음
    mock_cap.read.return_value = (False, None)
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(video_path, start_time, duration, frame_interval, target_size)

    assert "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다." in str(excinfo.value)

def test_download_and_sample_video_local_resize_failure(mocker):
    video_path = "/fake/video.mp4"
    start_time = 0
    duration = 60
    frame_interval = 1
    target_size = (256, 256)

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 1800]  # FPS, 총 프레임 수
    # Mock read() 메서드: 정상적인 프레임 반환
    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(1800)]
    mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    # cv2.resize을 모킹하여 예외 발생
    mocker.patch("cv2.resize", side_effect=Exception("Resize error"))

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(video_path, start_time, duration, frame_interval, target_size)

    assert "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다." in str(excinfo.value)
