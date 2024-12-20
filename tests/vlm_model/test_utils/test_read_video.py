# tests/vlm_model/test_utils/test_read_video.py

import pytest
import numpy as np
from unittest import mock
from vlm_model.utils.read_video import read_video_opencv
from vlm_model.exceptions import VideoProcessingError
import cv2

def test_read_video_opencv_success(mocker):
    video_path = "/fake/video.mp4"
    frame_indices = [10, 20, 30]
    sample_frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in frame_indices]

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    # read() 메서드 모킹: 40 프레임을 반환 (0~39)
    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(40)]
    mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # cv2.VideoCapture.read 호출 시 특정 프레임들을 반환하도록 설정
    result = read_video_opencv(video_path, frame_indices)

    assert len(result) == len(frame_indices)
    for i, frame in enumerate(result):
        assert np.array_equal(frame, frames[frame_indices[i]])

def test_read_video_opencv_video_not_found(mocker):
    video_path = "/fake/nonexistent_video.mp4"
    frame_indices = [0, 1, 2]

    # cv2.VideoCapture를 모킹하여 파일을 열 수 없다고 설정
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        read_video_opencv(video_path, frame_indices)

    assert f"비디오 파일을 열 수 없습니다: {video_path}" in str(excinfo.value)

def test_read_video_opencv_no_frames_found(mocker):
    video_path = "/fake/video.mp4"
    frame_indices = [100, 200, 300]

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    # read() 메서드 모킹: 50 프레임을 반환
    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(50)]
    mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        read_video_opencv(video_path, frame_indices)

    assert "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없음." in str(excinfo.value)

def test_read_video_opencv_unexpected_exception(mocker):
    video_path = "/fake/video.mp4"
    frame_indices = [10, 20, 30]

    # cv2.VideoCapture를 모킹
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    # read() 메서드 모킹하여 예외 발생
    mock_cap.read.side_effect = Exception("Unexpected error")
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        read_video_opencv(video_path, frame_indices)

    assert "비디오에서 프레임 추출 중 서버 오류가 발생했습니다." in str(excinfo.value)
