# vlm_model/routers/upload_video.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import os
from vlm_model.schemas.feedback import FeedbackResponse
import logging

router = APIRouter()

# 로깅 설정
logger = logging.getLogger(__name__)

# 업로드된 비디오를 저장할 디렉토리 설정
UPLOAD_DIR = "storage/input_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/receive-video/", response_model=FeedbackResponse)
async def receive_video_endpoint(file: UploadFile = File(...)):
    """
    클라이언트로부터 비디오 파일을 업로드 받아 서버에 저장합니다.
    """
    # 지원하는 파일 형식 확인
    ALLOWED_EXTENSIONS = {"webm", "mp4", "mov", "avi", "mkv"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"지원하지 않는 파일 형식: {file_extension}")
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 파일 저장 경로 설정
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # 비디오 파일 저장 (비동기 방식)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"비디오 파일 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"파일 저장 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {e}")

    # 피드백 응답 초기화 (빈 리스트로 반환)
    return FeedbackResponse(
        feedbacks=[],
        message="비디오 업로드 완료."
    )
