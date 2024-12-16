# vlm_model/routers/send_feedback.py

from fastapi import APIRouter, HTTPException
import os
import re
import uuid
import base64

from pathlib import Path

from vlm_model.schemas.feedback import FeedbackResponse
from vlm_model.utils.processing_video import process_video
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError, MediapipeHandlingError
from vlm_model.config import FEEDBACK_DIR, UPLOAD_DIR

import logging
import logging.config
import json

router = APIRouter()

logger = logging.getLogger(__name__)  # 'vlm_model.routers.send_feedback' 로거 사용

@router.get("/video-send-feedback/{video_id}/", response_model=FeedbackResponse)
async def send_feedback_endpoint(video_id: str):
    """
    video_id를 통해 저장된 비디오 파일을 처리하고 피드백 데이터를 반환합니다.
    """
    # VP9 변환된 비디오 파일 경로 확인
    video_path = UPLOAD_DIR / f"{video_id}_vp9.webm"

    # 변환된 VP9 파일이 없을 경우 원본 파일을 찾기 위해 확장자 목록 확인
    if not os.path.exists(video_path):
        video_extensions = ["webm", "mp4", "mov", "avi", "mkv"]
        video_path = None
        for ext in video_extensions:
            potential_path = os.path.join(UPLOAD_DIR, f"{video_id}.{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break

    # 비디오 파일을 찾지 못했을 경우 오류 반환
    if not video_path:
        logger.error(f"비디오 파일을 찾을 수 없습니다: video_id={video_id}", extra={
            "errorType": "FileNotFoundError",
            "error_message": f"video_id={video_id}"
        })
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")

    # 비디오 처리하여 피드백 생성
    try:
        feedback_data = process_video(video_path, video_id)
    except VideoProcessingError as vpe:
        logger.error(f"비디오 처리 중 오류 발생: {vpe.message}", extra={
            "errorType": "VideoProcessingError",
            "error_message": vpe.message
        })
        raise HTTPException(status_code=500, detail="비디오 처리 중 오류가 발생했습니다.") from vpe
    except ImageEncodingError as iee:
        logger.error(f"이미지 인코딩 중 오류 발생: {iee.message}", extra={
            "errorType": "ImageEncodingError",
            "error_message": iee.message
        })
        raise HTTPException(status_code=500, detail="이미지 인코딩 중 오류가 발생했습니다.") from iee
    except HTTPException as he:
        # 이미 HTTPException이 발생했으므로 다시 던짐
        raise he
    except Exception as e:
        logger.error(f"비디오 처리 중 예상치 못한 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise HTTPException(status_code=500, detail="비디오 처리 중 예상치 못한 오류가 발생했습니다.") from e

    # 피드백 데이터 확인 - 정상 처리
    if not feedback_data:
        logger.info(f"분석 결과 피드백할 내용이 없습니다: video_id={video_id}", extra={
            "errorType": "NoFeedback",
            "error_message": "분석 결과 피드백할 내용이 없습니다."
        })
        return FeedbackResponse(
            feedbacks=[],
            message="분석 결과 피드백할 내용이 없습니다.",
            problem="no_feedback"
        )

    logger.info(f"비디오 ID {video_id}에 대한 분석이 성공적으로 완료되었습니다.")
    return FeedbackResponse(
        feedbacks=feedback_data,
        message="피드백 데이터 생성 완료",
        problem=None
    )