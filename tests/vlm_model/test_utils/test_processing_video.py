# tests/vlm_model/test_utils/test_processing_video.py

import pytest
from unittest import mock
from vlm_model.utils.processing_video import process_video
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError
from fastapi import HTTPException

def test_process_video_success(mocker):
    file_path = "/fake/video.mp4"
    video_id = "test_video_id"

    # get_video_duration을 모킹하여 성공적으로 반환
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)

    # download_and_sample_video_local을 모킹하여 프레임 반환
    sample_frames = [mock.Mock() for _ in range(5)]
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=sample_frames)

    # analyze_frame을 모킹하여 정상적으로 결과 반환
    mediapipe_results = [
        {"posture_score": 0.9, "gaze_score": 0.8, "gestures_score": 0.7, "sudden_movement_score": 0.6},
        {"posture_score": 0.2, "gaze_score": 0.1, "gestures_score": 0.3, "sudden_movement_score": 0.4},
        {"posture_score": 0.85, "gaze_score": 0.75, "gestures_score": 0.65, "sudden_movement_score": 0.55},
        {"posture_score": 0.1, "gaze_score": 0.05, "gestures_score": 0.2, "sudden_movement_score": 0.3},
        {"posture_score": 0.95, "gaze_score": 0.85, "gestures_score": 0.75, "sudden_movement_score": 0.65},
    ]
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", side_effect=[mediapipe_results[i] for i in range(5)])

    # encode_feedback_image을 모킹하여 성공적으로 인코딩
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")

    # analyze_frames을 모킹하여 정상적인 피드백 반환
    feedback_data = [
        {
            "video_id": video_id,
            "frame_index": 1,
            "timestamp": "0m 10s",
            "feedback_text": {"feedback": {"posture": {"improvement": True}}},
            "image_base64": "encoded_image_string"
        },
        {
            "video_id": video_id,
            "frame_index": 3,
            "timestamp": "0m 30s",
            "feedback_text": {"feedback": {"posture": {"improvement": True}}},
            "image_base64": "encoded_image_string"
        }
    ]
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", return_value=([
        (mock.Mock(), 1, 1, 10.0),
        (mock.Mock(), 1, 3, 30.0)
    ], ["feedback1", "feedback2"]))

    # parse_feedback_text을 모킹하여 피드백 섹션 반환
    feedback_sections = mocker.Mock()
    feedback_sections.__fields__ = {"posture": mocker.Mock()}
    feedback_sections.posture.improvement = True
    mocker.patch("vlm_model.utils.processing_video.parse_feedback_text", return_value=feedback_sections)

    # encode_feedback_image를 모킹하여 Base64 문자열 반환
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")

    # os.path.exists을 모킹하여 FEEDBACK_DIR 존재 확인
    mocker.patch("os.path.exists", return_value=True)

    # os.path.join을 모킹하여 경로 반환
    mocker.patch("os.path.join", return_value="/fake/feedback_image.jpg")

    # open을 모킹하여 이미지 저장 성공
    mock_open = mocker.mock_open()
    mocker.patch("builtins.open", mock_open)

    # 함수 호출
    result = process_video(file_path, video_id)

    # 결과 확인
    assert len(result) == 2
    assert result[0]["video_id"] == video_id
    assert result[0]["frame_index"] == 1
    assert result[0]["timestamp"] == "0m 10s"
    assert result[0]["feedback_text"] == {"feedback": {"posture": {"improvement": True}}}
    assert result[0]["image_base64"] == "encoded_image_string"

def test_process_video_get_video_duration_failure(mocker):
    file_path = "/fake/video.mp4"
    video_id = "test_video_id"

    # get_video_duration을 모킹하여 예외 발생
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", side_effect=VideoProcessingError("Failed to get duration"))

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(file_path, video_id)

    assert "비디오 파일을 가져올 수 없습니다." in str(excinfo.value)

def test_process_video_download_failure(mocker):
    file_path = "/fake/video.mp4"
    video_id = "test_video_id"

    # get_video_duration을 모킹하여 성공적으로 반환
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)

    # download_and_sample_video_local을 모킹하여 예외 발생
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", side_effect=VideoProcessingError("Download failed"))

    with pytest.raises(VideoProcessingError) as excinfo:
        process_video(file_path, video_id)

    assert "프레임 추출 실패" in str(excinfo.value)

def test_process_video_analyze_frames_failure(mocker):
    file_path = "/fake/video.mp4"
    video_id = "test_video_id"

    # get_video_duration을 모킹하여 성공적으로 반환
    mocker.patch("vlm_model.utils.processing_video.get_video_duration", return_value=120.0)

    # download_and_sample_video_local을 모킹하여 프레임 반환
    sample_frames = [mock.Mock() for _ in range(5)]
    mocker.patch("vlm_model.utils.processing_video.download_and_sample_video_local", return_value=sample_frames)

    # analyze_frame을 모킹하여 정상적으로 결과 반환
    mediapipe_results = [
        {"posture_score": 0.9, "gaze_score": 0.8, "gestures_score": 0.7, "sudden_movement_score": 0.6},
    ]
    mocker.patch("vlm_model.utils.processing_video.analyze_frame", side_effect=[mediapipe_results[0]])

    # encode_feedback_image을 모킹하여 성공적으로 인코딩
    mocker.patch("vlm_model.utils.processing_video.encode_feedback_image", return_value="encoded_image_string")

    # analyze_frames을 모킹하여 예외 발생
    mocker.patch("vlm_model.utils.processing_video.analyze_frames", side_effect=Exception("Analyze frames failed"))

    with pytest.raises(HTTPException) as excinfo:
        process_video(file_path, video_id)

    assert "프레임 분석 중 오류가 발생했습니다." in str(excinfo.value)
