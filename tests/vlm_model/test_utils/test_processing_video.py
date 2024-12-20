import pytest
from unittest.mock import patch, MagicMock
from vlm_model.utils.processing_video import process_video
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError
from fastapi import HTTPException

@pytest.fixture
def test_video_path():
    return "/fake/video.mp4"

@pytest.fixture
def test_video_id():
    return "test_video_id"

def test_process_video_success(mocker, test_video_path, test_video_id):
    # get_video_duration Mock
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)

    # download_and_sample_video_local Mock: 프레임 5개 반환
    frames = [MagicMock() for _ in range(5)]
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=frames)

    # analyze_frame Mock: mediapipe 결과 반환
    # 문제 프레임 판단 조건: posture_score > 0.8 등
    # 여기서는 두 번째 프레임만 문제 프레임 가정
    def analyze_frame_side_effect(frame, prev_pose, prev_hand):
        # 첫 프레임: posture_score 0.9 -> 문제 프레임
        # 나머지 0.1로 문제 없음
        if frame is frames[0]:
            return {"posture_score":0.9,"gaze_score":0.0,"gestures_score":0.0,"sudden_movement_score":0.0}, None, None
        else:
            return {"posture_score":0.1,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1}, None, None
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", side_effect=analyze_frame_side_effect)

    # analyze_frames Mock
    def analyze_frames_side_effect(*args, **kwargs):
        # 문제 프레임 1개
        return [
            (frames[0],1,1,10.0)  # segment_number=1, frame_number=1, timestamp=10.0
        ], ["feedback_text"]
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", side_effect=analyze_frames_side_effect)

    # encode_feedback_image Mock
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")

    # parse_feedback_text Mock
    mock_sections = MagicMock()
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", return_value=mock_sections)

    # os.path.exists를 True로 Mock (FEEDBACK_DIR 존재)
    mocker.patch("os.path.exists", return_value=True)

    # os.path.join Mock
    mocker.patch("os.path.join", return_value="/fake/feedback_image.jpg")

    mock_open = mocker.mock_open()
    mocker.patch("builtins.open", mock_open)

    result = process_video(test_video_path, test_video_id)
    # 피드백 데이터가 1개 들어올 것
    assert len(result) == 1
    assert result[0]["video_id"] == test_video_id

def test_process_video_no_frames(mocker, test_video_path, test_video_id):
    # get_video_duration Mock
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)

    # download_and_sample_video_local이 None 또는 빈 리스트 반환
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=None)

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "비디오 파일에 문제가 있을 수 있습니다." in str(excinfo.value)

def test_process_video_download_failure(mocker, test_video_path, test_video_id):
    # download_and_sample_video_local 예외 발생
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", side_effect=VideoProcessingError("Download failed"))

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "프레임을 추출할 수 없습니다." in str(excinfo.value)

def test_process_video_analyze_frames_failure(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    # analyze_frame 문제 없는 프레임 반환
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.1,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1}, None, None))
    # 문제 프레임 없으므로 analyze_frames 호출
    def analyze_frames_side_effect(*args, **kwargs):
        raise Exception("Analyze frames failed")
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", side_effect=analyze_frames_side_effect)

    with pytest.raises(HTTPException) as excinfo:
        process_video(test_video_path, test_video_id)
    assert excinfo.value.status_code == 422
    assert "프레임 분석 중 오류" in str(excinfo.value)

def test_process_video_image_save_failure(mocker, test_video_path, test_video_id):
    # 문제 프레임이 존재하는 상황 가정
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.9,"gaze_score":0.0,"gestures_score":0.0,"sudden_movement_score":0.0}, None, None))
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", return_value=([(MagicMock(),1,1,10.0)], ["feedback1"]))
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")
    mock_sections = MagicMock()
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", return_value=mock_sections)
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.join", return_value="/fake/feedback_image.jpg")

    # open을 Mock
    mock_open = mocker.mock_open()
    mocker.patch("builtins.open", mock_open)
    # 이미지 저장 후 파일이 존재하지 않는다고 가정
    mocker.patch("os.path.exists", return_value=False)

    with pytest.raises(HTTPException) as excinfo:
        process_video(test_video_path, test_video_id)
    assert excinfo.value.status_code == 500
    assert "이미지 저장 중 오류가 발생했습니다." in str(excinfo.value)
