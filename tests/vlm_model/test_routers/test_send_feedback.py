# tests/vlm_model/test_routers/test_send_feedback.py

import pytest
from fastapi.testclient import TestClient
from unittest import mock
from pathlib import Path
from vlm_model.routers.send_feedback import router
from fastapi import FastAPI
from vlm_model.schemas.feedback import FeedbackResponse

# FastAPI 앱에 라우터를 포함시킴
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_send_feedback_success(client, mocker):
    video_id = "test_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    mocker.patch("vlm_model.routers.send_feedback.UPLOAD_DIR", Path("/fake/upload_dir"))
    mocker.patch("vlm_model.routers.send_feedback.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # 원본 비디오 파일 존재 모킹
    original_file = Path(f"/fake/upload_dir/{video_id}_original.mp4")
    mocker.patch("os.path.exists", side_effect=lambda path: path == original_file or path == Path(f"/fake/upload_dir/{video_id}_vp9.webm"))

    # Path.glob을 모킹하여 원본 비디오 파일 반환
    mocker.patch("vlm_model.routers.send_feedback.Path.glob", return_value=[original_file])

    # 코덱 변환 함수 모킹
    mock_convert = mocker.patch("vlm_model.routers.send_feedback.convert_to_vp9_if_needed", return_value=True)

    # 비디오 처리 함수 모킹
    feedback_data = [{"feedback": "Good job"}]
    mock_process = mocker.patch("vlm_model.routers.send_feedback.process_video", return_value=feedback_data)

    response = client.get(f"/video-send-feedback/{video_id}/")
    assert response.status_code == 200
    assert response.json() == {
        "feedbacks": feedback_data,
        "message": "피드백 데이터 생성 완료",
        "problem": None
    }

    # 함수 호출 검증
    mock_convert.assert_called_once()
    mock_process.assert_called_once_with(str(original_file), video_id)

def test_send_feedback_original_not_found(client, mocker):
    video_id = "nonexistent_video_id"

    # UPLOAD_DIR를 모킹
    mocker.patch("vlm_model.routers.send_feedback.UPLOAD_DIR", Path("/fake/upload_dir"))

    # 원본 비디오 파일 존재하지 않음 모킹
    mocker.patch("os.path.exists", return_value=False)

    response = client.get(f"/video-send-feedback/{video_id}/")
    assert response.status_code == 404
    assert response.json() == {"detail": "원본 비디오 파일을 찾을 수 없습니다."}

def test_send_feedback_codec_conversion_failure(client, mocker):
    video_id = "test_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    mocker.patch("vlm_model.routers.send_feedback.UPLOAD_DIR", Path("/fake/upload_dir"))
    mocker.patch("vlm_model.routers.send_feedback.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # 원본 비디오 파일 존재 모킹
    original_file = Path(f"/fake/upload_dir/{video_id}_original.mp4")
    mocker.patch("os.path.exists", side_effect=lambda path: path == original_file)

    # 코덱 변환 함수 모킹하여 실패
    mocker.patch("vlm_model.routers.send_feedback.convert_to_vp9_if_needed", side_effect=Exception("코덱 변환 오류"))

    response = client.get(f"/video-send-feedback/{video_id}/")
    assert response.status_code == 500
    assert response.json() == {"detail": "비디오 코덱 변환 중 오류가 발생했습니다."}

def test_send_feedback_video_processing_error(client, mocker):
    video_id = "test_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    mocker.patch("vlm_model.routers.send_feedback.UPLOAD_DIR", Path("/fake/upload_dir"))
    mocker.patch("vlm_model.routers.send_feedback.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # 원본 비디오 파일 존재 모킹
    original_file = Path(f"/fake/upload_dir/{video_id}_original.mp4")
    mocker.patch("os.path.exists", side_effect=lambda path: path == original_file or path == Path(f"/fake/upload_dir/{video_id}_vp9.webm"))

    # Path.glob을 모킹하여 원본 비디오 파일 반환
    mocker.patch("vlm_model.routers.send_feedback.Path.glob", return_value=[original_file])

    # 코덱 변환 함수 모킹
    mock_convert = mocker.patch("vlm_model.routers.send_feedback.convert_to_vp9_if_needed", return_value=True)

    # 비디오 처리 함수 모킹하여 예외 발생
    mock_process = mocker.patch("vlm_model.routers.send_feedback.process_video", side_effect=Exception("비디오 처리 오류"))

    response = client.get(f"/video-send-feedback/{video_id}/")
    assert response.status_code == 500
    assert response.json() == {"detail": "비디오 처리 중 예상치 못한 오류가 발생했습니다."}

    # 함수 호출 검증
    mock_convert.assert_called_once()
    mock_process.assert_called_once_with(str(original_file), video_id)
