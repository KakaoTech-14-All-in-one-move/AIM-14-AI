import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from vlm_model.utils.analysis import analyze_frames
from vlm_model.exceptions import PromptImportingError
from fastapi import HTTPException

@pytest.fixture
def dummy_frames():
    # 3개의 더미 프레임 (320x240 RGB)
    return [np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(3)]

@pytest.fixture
def dummy_timestamps():
    # 각 프레임에 대한 타임스탬프 (초 단위)
    return [10.0, 20.0, 30.0]

@pytest.fixture
def dummy_mediapipe_results():
    return [
        {"posture_score":0.9, "gaze_score":0.1, "gestures_score":0.2, "sudden_movement_score":0.3},
        {"posture_score":0.1, "gaze_score":0.8, "gestures_score":0.3, "sudden_movement_score":0.4},
        {"posture_score":0.1, "gaze_score":0.1, "gestures_score":0.1, "sudden_movement_score":0.8}
    ]

def test_analyze_frames_success(mocker, dummy_frames, dummy_timestamps, dummy_mediapipe_results):
    # Mock load_user_prompt
    mocker.patch("vlm_model.utils.analysis.load_user_prompt", return_value="User prompt content")

    # Mock encode_image
    mocker.patch("vlm_model.utils.analysis.encode_image", return_value="base64encodedimage")

    # Mock openai client
    mock_client = mocker.patch("vlm_model.utils.analysis.client.chat.completions.create")
    # OpenAI 응답 Mock
    mock_client.return_value.choices = [MagicMock(message=MagicMock(content='{"problem":"none"}'))]

    # parse_feedback_text Mock
    mocker.patch("vlm_model.utils.analysis.parse_feedback_text", return_value=MagicMock(**{"__fields__":["posture_body"], "posture_body":MagicMock(improvement=False)}))

    problematic_frames, feedbacks = analyze_frames(
        frames=dummy_frames,
        timestamps=dummy_timestamps,
        mediapipe_results=dummy_mediapipe_results,
        segment_idx=0,
        duration=60,
        segment_length=60,
        system_instruction="System instruction text",
        frame_interval=1
    )

    # 문제 프레임은 각 mediapipe 결과에서 점수가 0.8 초과할 때 발생하므로 dummy_mediapipe_results에 따라 판단
    # 첫 번째 프레임 posture_score=0.9 -> 문제 프레임
    # 두 번째 프레임 gaze_score=0.8 -> 문제 프레임
    # 세 번째 프레임 sudden_movement_score=0.8 -> 문제 프레임
    # 세 프레임 모두 문제 프레임이므로
    assert len(problematic_frames) == 3
    assert len(feedbacks) == 3

def test_analyze_frames_empty_mediapipe(mocker, dummy_frames, dummy_timestamps):
    # mediapipe_results가 비어있는 경우
    with pytest.raises(ValueError):
        analyze_frames(
            frames=dummy_frames,
            timestamps=dummy_timestamps,
            mediapipe_results=[],
            segment_idx=0,
            duration=60,
            segment_length=60,
            system_instruction="System instruction text"
        )

def test_analyze_frames_length_mismatch(mocker, dummy_frames):
    # mediapipe_results 길이가 frames와 다르면 ValueError
    with pytest.raises(ValueError):
        analyze_frames(
            frames=dummy_frames,
            timestamps=[10.0,20.0,30.0],
            mediapipe_results=[{}],  # 길이 불일치
            segment_idx=0,
            duration=60,
            segment_length=60,
            system_instruction="System instruction text"
        )

def test_analyze_frames_openai_error(mocker, dummy_frames, dummy_timestamps, dummy_mediapipe_results):
    mocker.patch("vlm_model.utils.analysis.load_user_prompt", return_value="User prompt")
    mocker.patch("vlm_model.utils.analysis.encode_image", return_value="base64encodedimage")

    mock_client = mocker.patch("vlm_model.utils.analysis.client.chat.completions.create", side_effect=ValueError("JSON 디코딩 오류"))

    with pytest.raises(HTTPException) as excinfo:
        analyze_frames(
            frames=dummy_frames,
            timestamps=dummy_timestamps,
            mediapipe_results=dummy_mediapipe_results,
            segment_idx=0,
            duration=60,
            segment_length=60,
            system_instruction="System instruction text"
        )
    assert excinfo.value.status_code == 400
    assert "피드백 파싱 과정 중 오류" in str(excinfo.value)
