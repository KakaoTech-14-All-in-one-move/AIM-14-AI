# tests/vlm_model/test_routers/test_upload_video.py

import pytest
from fastapi.testclient import TestClient
from unittest import mock
from pathlib import Path
from vlm_model.routers.upload_video import router
from fastapi import FastAPI
from vlm_model.schemas.feedback import UploadResponse

# FastAPI 앱에 라우터를 포함시킴
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_receive_video_success(client, mocker):
    # UPLOAD_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.upload_video.UPLOAD_DIR", Path("/fake/upload_dir"))

    # uuid.uuid4를 모킹하여 고정된 video_id 반환
    mock_uuid = mocker.patch("vlm_model.routers.upload_video.uuid.uuid4", return_value=mock.Mock(hex="test_video_id"))

    # shutil.copyfileobj를 모킹하여 파일 복사 성공
    mock_copy = mocker.patch("shutil.copyfileobj", return_value=None)

    # os.path.exists를 모킹하여 파일이 저장되었다고 가정
    mocker.patch("os.path.exists", return_value=True)

    # os.path.getsize를 모킹하여 파일 크기 반환
    mocker.patch("os.path.getsize", return_value=1024)

    # 샘플 비디오 파일 생성
    sample_file_content = b"fake video data"

    response = client.post(
        "/receive-video/",
        files={"file": ("sample_video.mp4", sample_file_content, "video/mp4")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": "test_video_id",
        "message": "비디오 업로드 완료. 피드백 데이터를 받으려면 /video-send-feedback/test_video_id/ 엔드포인트를 호출하세요."
    }

    # uuid4와 shutil.copyfileobj가 올바르게 호출되었는지 확인
    mock_uuid.assert_called_once()
    mock_copy.assert_called_once()

def test_receive_video_unsupported_file_type(client, mocker):
    # 샘플 비디오 파일 생성 (지원하지 않는 형식)
    sample_file_content = b"fake data"

    response = client.post(
        "/receive-video/",
        files={"file": ("sample_file.txt", sample_file_content, "text/plain")}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "지원하지 않는 파일 형식입니다."}

def test_receive_video_file_not_saved(client, mocker):
    # UPLOAD_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.upload_video.UPLOAD_DIR", Path("/fake/upload_dir"))

    # uuid.uuid4를 모킹하여 고정된 video_id 반환
    mock_uuid = mocker.patch("vlm_model.routers.upload_video.uuid.uuid4", return_value=mock.Mock(hex="test_video_id"))

    # shutil.copyfileobj를 모킹하여 파일 복사 성공
    mock_copy = mocker.patch("shutil.copyfileobj", return_value=None)

    # os.path.exists를 모킹하여 파일이 저장되지 않았다고 가정
    mocker.patch("os.path.exists", return_value=False)

    # os.remove를 모킹
    mock_remove = mocker.patch("os.remove", return_value=None)

    # 샘플 비디오 파일 생성
    sample_file_content = b"fake video data"

    response = client.post(
        "/receive-video/",
        files={"file": ("sample_video.mp4", sample_file_content, "video/mp4")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "파일이 저장되지 않았습니다."}

    # uuid4와 shutil.copyfileobj, os.remove가 올바르게 호출되었는지 확인
    mock_uuid.assert_called_once()
    mock_copy.assert_called_once()
    mock_remove.assert_called_once_with(Path("/fake/upload_dir/test_video_id_original.mp4"))

def test_receive_video_io_error(client, mocker):
    # UPLOAD_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.upload_video.UPLOAD_DIR", Path("/fake/upload_dir"))

    # uuid.uuid4를 모킹하여 고정된 video_id 반환
    mock_uuid = mocker.patch("vlm_model.routers.upload_video.uuid.uuid4", return_value=mock.Mock(hex="test_video_id"))

    # shutil.copyfileobj를 모킹하여 IOError 발생
    mocker.patch("shutil.copyfileobj", side_effect=IOError("파일 복사 오류"))

    # 샘플 비디오 파일 생성
    sample_file_content = b"fake video data"

    response = client.post(
        "/receive-video/",
        files={"file": ("sample_video.mp4", sample_file_content, "video/mp4")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "파일 저장 중 오류가 발생했습니다."}
