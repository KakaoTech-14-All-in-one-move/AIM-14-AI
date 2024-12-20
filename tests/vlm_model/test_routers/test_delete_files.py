# tests/vlm_model/test_routers/test_delete_files.py

import pytest
from fastapi.testclient import TestClient
from unittest import mock
from pathlib import Path
from vlm_model.routers.delete_files import router
from fastapi import FastAPI
from vlm_model.schemas.feedback import DeleteResponse

# FastAPI 앱에 라우터를 포함시킴
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_delete_files_success(client, mocker):
    video_id = "test_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir"))
    feedback_dir = mocker.patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # 삭제할 파일 목록 모킹
    input_files = [Path(f"/fake/upload_dir/{video_id}_1.mp4"), Path(f"/fake/upload_dir/{video_id}_2.webm")]
    output_files = [Path(f"/fake/feedback_dir/{video_id}_feedback.jpg")]

    # Path.glob을 모킹하여 파일 목록 반환
    mocker.patch.object(Path, "glob", side_effect=[
        input_files,  # UPLOAD_DIR.glob
        output_files  # FEEDBACK_DIR.glob
    ])

    # 파일 삭제 (Path.unlink)을 모킹
    mock_unlink = mocker.patch.object(Path, "unlink", return_value=None)

    response = client.delete(f"/delete_files/{video_id}")
    assert response.status_code == 200
    assert response.json() == {
        "video_id": video_id,
        "message": f"{video_id}와 관련된 파일 삭제에 성공했습니다."
    }

    # unlink가 올바르게 호출되었는지 검증
    assert mock_unlink.call_count == len(input_files) + len(output_files)
    for file in input_files + output_files:
        mock_unlink.assert_any_call()

def test_delete_files_no_files(client, mocker):
    video_id = "nonexistent_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir"))
    feedback_dir = mocker.patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # Path.glob을 모킹하여 빈 리스트 반환
    mocker.patch.object(Path, "glob", return_value=[])

    response = client.delete(f"/delete_files/{video_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "해당 video_id와 관련된 파일을 찾을 수 없습니다."}

def test_delete_files_unlink_failure(client, mocker):
    video_id = "test_video_id"

    # UPLOAD_DIR와 FEEDBACK_DIR을 모킹
    upload_dir = mocker.patch("vlm_model.routers.delete_files.UPLOAD_DIR", Path("/fake/upload_dir"))
    feedback_dir = mocker.patch("vlm_model.routers.delete_files.FEEDBACK_DIR", Path("/fake/feedback_dir"))

    # 삭제할 파일 목록 모킹
    input_files = [Path(f"/fake/upload_dir/{video_id}_1.mp4")]
    output_files = [Path(f"/fake/feedback_dir/{video_id}_feedback.jpg")]

    # Path.glob을 모킹하여 파일 목록 반환
    mocker.patch.object(Path, "glob", side_effect=[
        input_files,  # UPLOAD_DIR.glob
        output_files  # FEEDBACK_DIR.glob
    ])

    # 파일 삭제 (Path.unlink)을 모킹하여 첫 번째 파일 삭제는 성공, 두 번째는 실패
    def unlink_side_effect():
        yield None
        raise Exception("삭제 오류")

    mocker.patch.object(Path, "unlink", side_effect=[None, Exception("삭제 오류")])

    response = client.delete(f"/delete_files/{video_id}")
    assert response.status_code == 500
    assert response.json() == {"detail": f"{output_files[0].name} 파일 삭제에 실패했습니다."}
