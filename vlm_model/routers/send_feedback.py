# vlm_model/routers/send_feedback.py

from fastapi import APIRouter, HTTPException
import os
import re
import uuid
import base64

from vlm_model.schemas.feedback import (
    FeedbackSections,
    FeedbackDetails,
    FeedbackResponse,
    FeedbackFrame
)

from vlm_model.utils.download_video import download_and_sample_video_local
from vlm_model.utils.analysis import analyze_frames, parse_feedback_text
from vlm_model.utils.encoding_image import encode_image
from vlm_model.utils.video_duration import get_video_duration
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError
from vlm_model.openai_config import SYSTEM_INSTRUCTION
from vlm_model.config import FEEDBACK_DIR, UPLOAD_DIR

import logging
import logging.config
import json
from pathlib import Path

router = APIRouter()

logger = logging.getLogger(__name__)  # 'vlm_model.routers.send_feedback' 로거 사용

def process_video(file_path: str):
    """
    비디오 파일을 처리하여 피드백 데이터를 생성합니다.
    """
    try:
        video_duration = get_video_duration(file_path)
    except VideoProcessingError as vpe:
        logger.error(f"비디오 파일을 가져올 수 없습니다: {file_path} - {vpe.message}", extra={
            "errorType": "VideoProcessingError",
            "error_message": f"비디오 파일을 가져올 수 없습니다: {file_path} - {vpe.message}"
        })
        raise VideoProcessingError("비디오 파일을 가져올 수 없습니다.") from vpe

    # 세그먼트 길이와 프레임 간격 설정
    segment_length = 60  # 초
    frame_interval = 2    # 초

    feedback_data = []

    # 디렉터리 경로가 지정되었는지 확인
    if FEEDBACK_DIR is None or not FEEDBACK_DIR.exists():
        logger.error("피드백 이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다.", extra={
            "errorType": "VideoProcessingError",
            "error_message": "피드백 이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다."
        })
        raise VideoProcessingError("피드백 이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다.")

    # 각 세그먼트별로 프레임 추출 및 피드백 분석
    for start_time in range(0, int(video_duration), segment_length):
        segment_index = start_time // segment_length
        try:
            frames = download_and_sample_video_local(file_path, start_time, segment_length, frame_interval)
        except VideoProcessingError as vpe:
            logger.error(f"프레임을 추출할 수 없습니다: {vpe.message}", extra={
                "errorType": "VideoProcessingError",
                "error_message": f"프레임 추출 실패: {vpe.message}"
            })
            raise VideoProcessingError("프레임을 추출할 수 없습니다.") from vpe

        if frames is None or len(frames) == 0:
            logger.error(f"프레임을 추출할 수 없습니다. 비디오 파일에 문제가 있을 수 있습니다: {file_path}", extra={
                "errorType": "VideoProcessingError",
                "error_message": f"비디오 파일에 문제가 있을 수 있습니다. {file_path}"
            })
            raise VideoProcessingError("비디오 파일에 문제가 있을 수 있습니다.")

        try:
            problematic_frames, feedbacks = analyze_frames(
                frames=frames,
                segment_idx=segment_index,
                duration=segment_length,
                segment_length=segment_length,
                system_instruction=SYSTEM_INSTRUCTION,
                frame_interval=frame_interval
            )
        except Exception as e:
            logger.error(f"프레임 분석 중 오류 발생: {e}",  extra={
                "errorType": type(e).__name__,
                "error_message": str(e)
            })
            raise HTTPException(status_code=422, detail="프레임 분석 중 오류가 발생했습니다.") from e

        logger.debug(f"프레임 수: {len(problematic_frames)}, 피드백 수: {len(feedbacks)}")

        for frame_info, feedback_text in zip(problematic_frames, feedbacks):
            frame, segment_number, frame_number, timestamp = frame_info

            # 이미지 인코딩 (Base64)
            try:
                image_base64 = encode_image(frame)
                if not image_base64:
                    logger.error(f"프레임 {frame_number}의 이미지 인코딩 실패", extra={
                        "errorType": "ImageEncodingError",
                        "error_message": f"이미지 인코딩 실패. 프레임 {frame_number}"
                    })
                    raise ImageEncodingError("이미지 인코딩이 실패했습니다.") from iee
            except ImageEncodingError as iee:
                logger.error(f"이미지 인코딩 실패: {iee.message}", extra={
                    "errorType": "ImageEncodingError",
                    "error_message": f"이미지 인코딩 실패: {iee.message}"
                })
                raise ImageEncodingError("이미지 인코딩 중 오류가 발생했습니다.") from iee

            # feedback_text를 FeedbackSections 구조로 변환
            try:
                feedback_sections = parse_feedback_text(feedback_text)
                logger.debug(f"변환된 피드백 섹션: {feedback_sections}")
            except VideoProcessingError as vpe:
                logger.error(f"피드백 텍스트 파싱 오류: {vpe.message}", extra={
                    "errorType": "VideoProcessingError",
                    "error_message": f"피드백 텍스트 파싱 오류: {vpe.message}"
                })
                raise VideoProcessingError("피드백 텍스트를 생성하는 중 오류가 발생했습니다.") from vpe
            except Exception as e:
                logger.error(f"피드백 텍스트 파싱 중 예상치 못한 오류 발생: {str(e)}", extra={
                    "errorType": type(e).__name__,
                    "error_message": str(e)
                })
                raise VideoProcessingError("피드백 텍스트를 생성하는 중 오류가 발생했습니다.") from e

            # 피드백 데이터 추가
            feedback_frame = FeedbackFrame(
                frame_index=frame_number,
                timestamp=timestamp,
                feedback_text=feedback_sections,
                image_base64=image_base64
            )
            feedback_data.append(feedback_frame.dict())

            # 피드백 이미지를 저장하는 경우
            if FEEDBACK_DIR:
                safe_timestamp = re.sub(r'[^\w_]', '', timestamp.replace("m ", "m_").replace(" ", "_").replace("s", "s_").strip("_"))
                unique_id = uuid.uuid4().hex  # 고유한 식별자 생성
                image_filename = f"{video_id}_segment_{segment_number}_frame_{frame_number}_{safe_timestamp}_{unique_id}.jpg"  # video_id 포함
                image_path = os.path.join(FEEDBACK_DIR, image_filename)
                try:
                    with open(image_path, "wb") as img_file:
                        img_file.write(base64.b64decode(image_base64))
                    if not os.path.exists(image_path):
                        raise IOError("이미지가 지정된 경로에 저장되지 않았습니다.")

                except IOError as ioe:
                    logger.error(f"이미지 저장 중 오류 발생: {ioe}", extra={
                        "errorType": "ImageSaveError",
                        "error_message": f"이미지 저장 중 오류 발생: {ioe}"
                    })
                    raise HTTPException(status_code=500, detail="이미지 저장 중 오류가 발생했습니다.") from ioe

                except Exception as e:
                    logger.error(f"이미지 저장 중 예상치 못한 오류 발생: {e}", extra={
                        "errorType": "ImageSaveError",
                        "error_message": f"이미지 저장 중 오류 발생: {e}"
                    })
                    raise HTTPException(status_code=500, detail="이미지 저장 중 오류가 발생했습니다.") from e

    # 피드백 데이터 반환 (비어 있을 수 있음)        
    return feedback_data

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
        feedback_data = process_video(video_path)
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