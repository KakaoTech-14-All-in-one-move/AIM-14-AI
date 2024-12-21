import pytest
from unittest.mock import patch, mock_open, MagicMock
from fastapi.testclient import TestClient
from vlm_model.routers.upload_video import router
from vlm_model.exceptions import VideoImportingError
from fastapi import FastAPI, UploadFile
import io

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_receive_video_success(client, mocker):
    # 모의 파일
    file_content = b"fake_video_data"
    upload_file = UploadFile(filename="test.mp4", file=io.BytesIO(file_content))

    # 파일 시스템 동작 Mock
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=len(file_content))

    m = mock_open()
    with patch("builtins.open", m):
        response = client.post("/receive-video/", files={"file": ("test.mp4", file_content, "video/mp4")})
        assert response.status_code == 200
        json_data = response.json()
        assert "video_id" in json_data
        assert "비디오 업로드 완료" in json_data["message"]

def test_receive_video_unsupported_format(client):
    # 지원하지 않는 형식
    file_content = b"fake_video_data"
    response = client.post("/receive-video/", files={"file": ("test.txt", file_content, "text/plain")})
    assert response.status_code == 400
    assert "지원하지 않는 파일 형식입니다." in response.text

def test_receive_video_file_not_saved(client, mocker):
    file_content = b"fake_video_data"
    # 파일 저장되었는지 확인할 때 os.path.exists가 False 반환
    mocker.patch("os.path.exists", return_value=False)

    m = mock_open()
    with patch("builtins.open", m):
        # 비디오 파일 형식은 mp4로 가정
        response = client.post("/receive-video/", files={"file": ("test.mp4", file_content, "video/mp4")})
        # 파일이 저장되지 않았으므로 VideoImportingError 발생해야 함
        # VideoImportingError는 try-except에서 HTTPException으로 변환되지 않았으므로
        # raise vie로 올라감 -> 500 Internal Server Error일 가능성
        assert response.status_code == 500
        assert "파일이 저장되지 않았습니다." in response.text

def test_receive_video_io_error(client, mocker):
    file_content = b"fake_video_data"

    # open 중 IOError 발생 시 模拟
    def open_side_effect(*args, **kwargs):
        raise IOError("I/O error")

    m = mock_open(side_effect=open_side_effect)
    with patch("builtins.open", m):
        response = client.post("/receive-video/", files={"file": ("test.mp4", file_content, "video/mp4")})
        # IOError 발생 시 500 에러
        assert response.status_code == 500
        assert "파일 저장 중 오류가 발생했습니다." in response.text

def test_receive_video_unknown_error(client, mocker):
    file_content = b"fake_video_data"

    def open_side_effect(*args, **kwargs):
        raise Exception("Unknown error")

    m = mock_open(side_effect=open_side_effect)
    with patch("builtins.open", m):
        response = client.post("/receive-video/", files={"file": ("test.mp4", file_content, "video/mp4")})
        assert response.status_code == 500
        assert "파일 처리 중 예기치 않은 오류가 발생했습니다." in response.text
