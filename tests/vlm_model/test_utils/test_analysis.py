# tests/vlm_model/test_utils/test_analysis.py

import pytest
import numpy as np
from unittest import mock
from vlm_model.utils.analysis import analyze_frames
from vlm_model.exceptions import VideoProcessingError
from fastapi import HTTPException

def test_analyze_frames_success(mocker):
    frames = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
    timestamps = [10.0, 20.0, 30.0, 40.0, 50.0]
    mediapipe_results = [
        {"posture_score": 0.9, "gaze_score": 0.8, "gestures_score": 0.7, "sudden_movement_score": 0.6},
        {"posture_score": 0.2, "gaze_score": 0.1, "gestures_score": 0.3, "sudden_movement_score": 0.4},
        {"posture_score": 0.85, "gaze_score": 0.75, "gestures_score": 0.65, "sudden_movement_score": 0.55},
        {"posture_score": 0.1, "gaze_score": 0.05, "gestures_score": 0.2, "sudden_movement_score": 0.3},
        {"posture_score": 0.95, "gaze_score": 0.85, "gestures_score": 0.75, "sudden_movement_score": 0.65},
    ]
    segment_idx = 0
    duration = 60
    segment_length = 60
    system_instruction = "System instruction text."

    # load_user_prompt을 모킹
    mocker.patch("vlm_model.utils.analysis.load_user_prompt", return_value="User prompt.")

    # encode_image을 모킹하여 항상 성공적으로 인코딩
    mocker.patch("vlm_model.utils.analysis.encode_image", return_value="encoded_image_string")

    # client.chat.completions.create를 모킹하여 성공적인 응답 반환
    mock_response = mocker.Mock()
    mock_response.choices = [mock.Mock(message=mock.Mock(content="```json\n{\"feedback\": {\"posture\": {\"improvement\": true}}}\n```"))]
    mocker.patch("vlm_model.utils.analysis.client.chat.completions.create", return_value=mock_response)

    # parse_feedback_text을 모킹하여 피드백 섹션 반환
    feedback_sections = mocker.Mock()
    feedback_sections.__fields__ = {"posture": mocker.Mock()}
    feedback_sections.posture.improvement = True
    mocker.patch("vlm_model.utils.analysis.parse_feedback_text", return_value=feedback_sections)

    # 함수 호출
    problematic_frames, feedbacks = analyze_frames(
        frames=frames,
        timestamps=timestamps,
        mediapipe_results=mediapipe_results,
        segment_idx=segment_idx,
        duration=duration,
        segment_length=segment_length,
        system_instruction=system_instruction,
        frame_interval=1
    )

    # 결과 확인
    assert len(problematic_frames) == 3  # frames 0, 2, 4
    assert len(feedbacks) == 3
    for feedback in feedbacks:
        assert feedback == "```json\n{\"feedback\": {\"posture\": {\"improvement\": true}}}\n```"

def test_analyze_frames_empty_mediapipe_results(mocker):
    frames = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)]
    timestamps = [10.0]
    mediapipe_results = []
    segment_idx = 0
    duration = 60
    segment_length = 60
    system_instruction = "System instruction text."

    with pytest.raises(ValueError) as excinfo:
        analyze_frames(
            frames=frames,
            timestamps=timestamps,
            mediapipe_results=mediapipe_results,
            segment_idx=segment_idx,
            duration=duration,
            segment_length=segment_length,
            system_instruction=system_instruction,
            frame_interval=1
        )

    assert "Mediapipe 결과가 비어 있습니다. 필수 입력값입니다." in str(excinfo.value)

def test_analyze_frames_mismatched_lengths(mocker):
    frames = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)]
    timestamps = [10.0]
    mediapipe_results = [
        {"posture_score": 0.9, "gaze_score": 0.8, "gestures_score": 0.7, "sudden_movement_score": 0.6},
        {"posture_score": 0.2, "gaze_score": 0.1, "gestures_score": 0.3, "sudden_movement_score": 0.4},
    ]
    segment_idx = 0
    duration = 60
    segment_length = 60
    system_instruction = "System instruction text."

    with pytest.raises(ValueError) as excinfo:
        analyze_frames(
            frames=frames,
            timestamps=timestamps,
            mediapipe_results=mediapipe_results,
            segment_idx=segment_idx,
            duration=duration,
            segment_length=segment_length,
            system_instruction=system_instruction,
            frame_interval=1
        )

    assert "mediapipe_results의 길이와 frames의 길이가 일치하지 않습니다." in str(excinfo.value)

def test_analyze_frames_openai_authentication_error(mocker):
    frames = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)]
    timestamps = [10.0]
    mediapipe_results = [
        {"posture_score": 0.9, "gaze_score": 0.8, "gestures_score": 0.7, "sudden_movement_score": 0.6}
    ]
    segment_idx = 0
    duration = 60
    segment_length = 60
    system_instruction = "System instruction text."

    # load_user_prompt을 모킹
    mocker.patch("vlm_model.utils.analysis.load_user_prompt", return_value="User prompt.")

    # encode_image을 모킹하여 성공적으로 인코딩
    mocker.patch("vlm_model.utils.analysis.encode_image", return_value="encoded_image_string")

    # client.chat.completions.create를 모킹하여 AuthenticationError 발생
    from openai import AuthenticationError
    mocker.patch("vlm_model.utils.analysis.client.chat.completions.create", side_effect=AuthenticationError("Invalid API key"))

    with pytest.raises(HTTPException) as excinfo:
        analyze_frames(
            frames=frames,
            timestamps=timestamps,
            mediapipe_results=mediapipe_results,
            segment_idx=segment_idx,
            duration=duration,
            segment_length=segment_length,
            system_instruction=system_instruction,
            frame_interval=1
        )

    assert excinfo.value.status_code == 401
    assert "인증 오류: API 키를 확인해주세요." in excinfo.value.detail
