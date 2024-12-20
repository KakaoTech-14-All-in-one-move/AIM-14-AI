# tests/vlm_model/test_routers/test_delete_files.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
from vlm_model.routers.delete_files import router
from fastapi import FastAPI

# FastAPI 앱에 라우터를 포함
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_delete_files_success(client):
    video_id = "test_video_id"

    with patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir")), \
         patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir")):

        upload_files = [
            Path(f"/fake/upload_dir/{video_id}_1.mp4"),
            Path(f"/fake/upload_dir/{video_id}_2.mkv")
        ]
        feedback_files = [
            Path(f"/fake/feedback_dir/{video_id}_feedback.jpg")
        ]

        with patch.object(Path, "glob", side_effect=[upload_files, feedback_files]):
            with patch.object(Path, "unlink", return_value=None):
                response = client.delete(f"/delete_files/{video_id}")
                assert response.status_code == 200
                assert response.json() == {
                    "video_id": video_id,
                    "message": f"{video_id}와 관련된 파일 삭제에 성공했습니다."
                }


def test_delete_files_no_files(client):
    video_id = "nonexistent_video_id"

    with patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir")), \
         patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir")):

        with patch.object(Path, "glob", return_value=[]):
            response = client.delete(f"/delete_files/{video_id}")
            assert response.status_code == 404
            assert response.json() == {"detail": "해당 video_id와 관련된 파일을 찾을 수 없습니다."}


def test_delete_files_unlink_failure(client):
    video_id = "test_video_id"

    with patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir")), \
         patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir")):

        upload_files = [
            Path(f"/fake/upload_dir/{video_id}_1.mp4")
        ]
        feedback_files = [
            Path(f"/fake/feedback_dir/{video_id}_feedback.jpg")
        ]

        with patch.object(Path, "glob", side_effect=[upload_files, feedback_files]):
            with patch.object(Path, "unlink", side_effect=[None, Exception("삭제 오류")]):
                response = client.delete(f"/delete_files/{video_id}")
                assert response.status_code == 500
                assert response.json() == {"detail": f"{feedback_files[0].name} 파일 삭제에 실패했습니다."}
