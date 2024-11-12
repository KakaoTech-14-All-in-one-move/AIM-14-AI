# vlm_model/routers/upload_video.py

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Response
import os
import uuid
import logging

from vlm_model.schemas.feedback import UploadResponse
from vlm_model.utils.video_duration import get_video_duration

router = APIRouter()

# 비디오 저장 경로 설정
UPLOAD_DIR = "storage/input_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 로깅 설정
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@router.post("/receive-video/", response_model=UploadResponse)
async def receive_video_endpoint(file: UploadFile = File(...), response: Response):
    """
    비디오 파일을 업로드 받아 저장하고, video_id를 반환합니다.
    """
    # 응답 헤더에 CORS 설정 추가
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "False"

    # 요청 수신 로그
    logger.info("receive_video_endpoint called")
    logger.info(f"Received file: {file.filename}")

    # 지원하는 파일 형식 확인
    ALLOWED_EXTENSIONS = {"webm", "mp4", "mov", "avi", "mkv"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"지원하지 않는 파일 형식: {file_extension}")
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 고유한 video_id 생성
    video_id = uuid.uuid4().hex

    # 비디오 파일 저장 경로 설정
    video_filename = f"{video_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, video_filename)

    # 비디오 파일 저장
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"비디오 파일 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"파일 저장 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {e}")

    return UploadResponse(
        video_id=video_id,
        message="비디오 업로드 완료. 피드백 데이터를 받으려면 /send-feedback/{video_id} 엔드포인트를 호출하세요."
    )
    

