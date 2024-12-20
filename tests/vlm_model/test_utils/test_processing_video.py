import pytest
from unittest.mock import patch, MagicMock, mock_open
from vlm_model.utils.processing_video import process_video
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError
from fastapi import HTTPException

@pytest.fixture
def test_video_path():
    return "/fake/video.mp4"

@pytest.fixture
def test_video_id():
    return "test_video_id"

def test_process_video_no_problem_frames(mocker, test_video_path, test_video_id):
    # get_video_duration Mock
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    # download_and_sample_video_local Mock: 60프레임
    frames = [MagicMock() for _ in range(60)]
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=frames)

    # analyze_frame: 문제 없는 프레임
    def no_problem_frame(*args,**kwargs):
        return {"posture_score":0.1,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1}, None, None
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", side_effect=no_problem_frame)

    # 문제 프레임 없으므로 analyze_frames 호출 안됨
    # encode_feedback_image, parse_feedback_text 필요 없음

    result = process_video(test_video_path, test_video_id)
    # 문제 프레임 없어 feedback_data empty
    assert len(result) == 0

def test_process_video_problem_frames(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    frames = [MagicMock() for _ in range(60)]
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=frames)

    def first_frame_problem(frame, ppose, phand):
        if frame is frames[0]:
            return {"posture_score":0.9,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1},None,None
        return {"posture_score":0.1,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1},None,None
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", side_effect=first_frame_problem)

    # analyze_frames 호출 (문제 프레임 1개)
    def analyze_frames_side_effect(*args,**kwargs):
        return [(frames[0],1,1,10.0)], ["feedback_text"]
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", side_effect=analyze_frames_side_effect)

    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")
    mock_sections = MagicMock()
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", return_value=mock_sections)

    # FEEDBACK_DIR 존재
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.join", return_value="/fake/feedback_image.jpg")
    m = mock_open()
    mocker.patch("builtins.open", m)

    result = process_video(test_video_path, test_video_id)
    assert len(result) == 1
    assert result[0]["video_id"] == test_video_id

def test_process_video_download_failure(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", side_effect=VideoProcessingError("Download failed"))

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "프레임을 추출할 수 없습니다." in str(excinfo.value)

def test_process_video_no_frames(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[])

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "비디오 파일에 문제가 있을 수 있습니다." in str(excinfo.value)

def test_process_video_analyze_frames_failure(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.9,"gaze_score":0.1,"gestures_score":0.1,"sudden_movement_score":0.1},None,None))
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", side_effect=Exception("Analyze frames failed"))

    with pytest.raises(HTTPException) as excinfo:
        process_video(test_video_path, test_video_id)
    assert excinfo.value.status_code == 422

def test_process_video_image_encoding_failure(mocker, test_video_path, test_video_id):
    # 문제 프레임 1개 시나리오
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.9,"gaze_score":0.0,"gestures_score":0.0,"sudden_movement_score":0.0},None,None))
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", return_value=([(MagicMock(),1,1,10.0)], ["feedback1"]))

    # encode_feedback_image 실패
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", side_effect=ImageEncodingError("Encode failed"))
    mocker.patch("os.path.exists", return_value=True)

    with pytest.raises(ImageEncodingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "이미지 인코딩 중 오류가 발생했습니다." in str(excinfo.value)

def test_process_video_feedback_parse_failure(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.9,"gaze_score":0.0,"gestures_score":0.0,"sudden_movement_score":0.0},None,None))
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", return_value=([(MagicMock(),1,1,10.0)], ["feedback1"]))
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")

    # parse_feedback_text 오류
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", side_effect=VideoProcessingError("Parse error"))
    mocker.patch("os.path.exists", return_value=True)

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(test_video_path, test_video_id)
    assert "피드백 텍스트를 생성하는 중 오류" in str(excinfo.value)

def test_process_video_image_save_failure(mocker, test_video_path, test_video_id):
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=[MagicMock()])
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", return_value=({"posture_score":0.9,"gaze_score":0.0,"gestures_score":0.0,"sudden_movement_score":0.0},None,None))
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", return_value=([(MagicMock(),1,1,10.0)], ["feedback1"]))
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")
    mock_sections = MagicMock()
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", return_value=mock_sections)

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.join", return_value="/fake/feedback_image.jpg")
    m = mock_open()
    mocker.patch("builtins.open", m)

    # os.path.exists가 이미지 저장 후 False
    def exists_side_effect(path):
        if "feedback_image.jpg" in path:
            return False
        return True
    mocker.patch("os.path.exists", side_effect=exists_side_effect)

    with pytest.raises(HTTPException) as excinfo:
        process_video(test_video_path, test_video_id)
    assert excinfo.value.status_code == 500
    assert "이미지 저장 중 오류가 발생했습니다." in str(excinfo.value)
