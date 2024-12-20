import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from vlm_model.utils.download_video import download_and_sample_video_local
from vlm_model.exceptions import VideoProcessingError

@pytest.fixture
def dummy_video_path():
    return "/fake/video.mp4"

def test_download_and_sample_zero_fps(mocker, dummy_video_path):
    # FPS=0인 경우 기본값 30.0 사용
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [0.0, 1800]  # FPS=0.0, total_frames=1800
    frames = [np.random.randint(0,256,(480,640,3),dtype=np.uint8) for _ in range(1800)]
    mock_cap.read.side_effect = [(True, f) for f in frames] + [(False,None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("cv2.resize", side_effect=lambda frame, size: frame)

    # 30fps로 간주하므로 60초 동안 1초에 1프레임 -> 60프레임 추출 기대
    result = download_and_sample_video_local(dummy_video_path, duration=60, frame_interval=1, target_size=(256,256))
    assert result.shape[0] == 60

def test_download_and_sample_no_total_frames(mocker, dummy_video_path):
    # 총 프레임 수를 0으로 반환하는 경우 (비정상 비디오)
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 0]  # FPS=30.0, total_frames=0
    mock_cap.read.return_value = (False,None)  # 프레임 없음
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(dummy_video_path)
    assert "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다." in str(excinfo.value)

def test_download_and_sample_resize_error(mocker, dummy_video_path):
    # resize 중 오류 발생
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 1800] 
    frames = [np.random.randint(0,256,(480,640,3),dtype=np.uint8) for _ in range(1800)]
    mock_cap.read.side_effect = [(True,f) for f in frames] + [(False,None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("cv2.resize", side_effect=Exception("Resize error"))

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(dummy_video_path)
    assert "비디오에서 프레임 추출 중 서버 오류가 발생했습니다." in str(excinfo.value)

def test_download_and_sample_unexpected_error(mocker, dummy_video_path):
    # VideoProcessingError가 아닌 다른 예외 발생
    mock_cap = mocker.Mock()
    mock_cap.isOpened.side_effect = Exception("Unexpected error in open")
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(dummy_video_path)
    assert "비디오에서 프레임 추출 중 서버 오류가 발생했습니다." in str(excinfo.value)
