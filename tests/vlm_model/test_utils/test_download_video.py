import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from vlm_model.utils.download_video import download_and_sample_video_local
from vlm_model.exceptions import VideoProcessingError

@pytest.fixture
def dummy_video_path():
    return "/fake/video.mp4"

def test_download_and_sample_success(mocker, dummy_video_path):
    # Mock VideoCapture
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    # FPS=30, 총프레임=1800 (60초 * 30fps), start_time=0, duration=60, frame_interval=1
    mock_cap.get.side_effect = [30.0, 1800]  
    # read() 메서드: 1800 프레임 정상 반환 후 False 반환
    frames = [np.random.randint(0,256,(480,640,3),dtype=np.uint8) for _ in range(1800)]
    mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("cv2.resize", side_effect=lambda frame, size: np.resize(frame,(size[1],size[0],3)))

    result = download_and_sample_video_local(dummy_video_path, start_time=0, duration=60, frame_interval=1, target_size=(256,256))
    # 매초 한 프레임씩 60초=60프레임 추출 기대(30fps에서 1초마다 1프레임 추출)
    assert result.shape[0] == 60
    assert result.shape[1:] == (256,256,3)

def test_download_and_sample_video_not_found(mocker, dummy_video_path):
    # 파일 열 수 없음
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(dummy_video_path)
    assert "비디오 파일을 열 수 없습니다" in str(excinfo.value)

def test_download_and_sample_no_frames_found(mocker, dummy_video_path):
    # 비디오 열리나, 요청한 프레임 못찾음
    mock_cap = mocker.Mock()
    mock_cap.isOpened.return_value = True
    # FPS=30, 총프레임=10이라고 가정(너무 짧은 비디오)
    mock_cap.get.side_effect = [30.0, 10]
    # read() 메서드도 10프레임 후 False
    frames = [np.random.randint(0,256,(480,640,3),dtype=np.uint8) for _ in range(10)]
    mock_cap.read.side_effect = [(True,f) for f in frames] + [(False,None)]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("cv2.resize", side_effect=lambda frame, size: frame)

    # 실제로는 60초 간격 1초마다 프레임 => 60프레임 필요하지만 비디오가 너무 짧음
    with pytest.raises(VideoProcessingError) as excinfo:
        download_and_sample_video_local(dummy_video_path, duration=60, frame_interval=1)
    assert "지정된 프레임 인덱스에 해당하는 프레임을 찾을 수 없습니다." in str(excinfo.value)

def test_download_and_sample_resize_failure(mocker, dummy_video_path):
    # resize 중 예외 발생
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
