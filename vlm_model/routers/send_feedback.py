# vlm_model/routers/send_feedback.py

from fastapi import APIRouter, HTTPException
import os
import re
import uuid
import logging
import base64
import json
from pathlib import Path

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
from vlm_model.config import SYSTEM_INSTRUCTION

router = APIRouter()

# 비디오 저장 경로와 피드백 저장 경로 설정
# Docker에서 쓰기 가능한 경로 설정
UPLOAD_DIR = Path('/tmp/storage/input_video')
FEEDBACK_DIR = Path('/tmp/storage/output_feedback_frame')

# 디렉토리가 존재하지 않으면 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# 로깅 설정
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def process_video(file_path: str):
    """
    비디오 파일을 처리하여 피드백 데이터를 생성합니다.
    """
    video_duration = get_video_duration(file_path)
    if video_duration is None or video_duration <= 0:
        logger.error(f"비디오 길이를 가져올 수 없습니다: {file_path}")
        return []

    # 세그먼트 길이와 프레임 간격 설정
    segment_length = 60  # 초
    frame_interval = 3    # 초

    feedback_data = []

    # 각 세그먼트별로 프레임 추출 및 피드백 분석
    for start_time in range(0, int(video_duration), segment_length):
        segment_index = start_time // segment_length

        frames = download_and_sample_video_local(file_path, start_time, segment_length, frame_interval)
        if frames is not None and len(frames) > 0:
            problematic_frames, feedbacks = analyze_frames(
                frames=frames,
                segment_idx=segment_index,
                duration=segment_length,
                segment_length=segment_length,
                system_instruction=SYSTEM_INSTRUCTION,
                frame_interval=frame_interval
            )

            logger.debug(f"프레임 수: {len(problematic_frames)}, 피드백 수: {len(feedbacks)}")

            for frame_info, feedback_text in zip(problematic_frames, feedbacks):
                frame, segment_number, frame_number, timestamp = frame_info

                # 이미지 인코딩 (Base64)
                image_base64 = encode_image(frame)
                if not image_base64:
                    logger.warning(f"프레임 {frame_number}의 이미지 인코딩 실패")
                    continue

                # feedback_text를 FeedbackSections 구조로 변환
                try:
                    feedback_sections = parse_feedback_text(feedback_text)
                    logger.debug(f"변환된 피드백 섹션: {feedback_sections}")
                except HTTPException as e:
                    logger.error(f"피드백 텍스트 파싱 오류: {str(e.detail)}")
                    continue
                except Exception as e:
                    logger.error(f"피드백 텍스트 파싱 중 예상치 못한 오류 발생: {str(e)}")
                    continue

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
                    image_filename = f"segment_{segment_number}_frame_{frame_number}_{safe_timestamp}_{unique_id}.jpg"
                    image_path = os.path.join(FEEDBACK_DIR, image_filename)
                    try:
                        with open(image_path, "wb") as img_file:
                            img_file.write(base64.b64decode(image_base64))
                    except Exception as e:
                        logger.error(f"이미지 저장 중 오류 발생: {e}")
        else:
            logger.warning(f"세그먼트 {segment_index+1}에서 프레임을 추출할 수 없습니다.")

    if not feedback_data:
        logger.info("피드백 데이터를 생성하지 못했습니다.")

    return feedback_data

@router.get("/video-send-feedback/{video_id}/", response_model=FeedbackResponse)
async def send_feedback_endpoint(video_id: str):
    """
    video_id를 통해 저장된 비디오 파일을 처리하고 피드백 데이터를 반환합니다.
    """
    # 저장된 비디오 파일 경로 확인
    video_extensions = ["webm", "mp4", "mov", "avi", "mkv"]
    video_path = None
    for ext in video_extensions:
        potential_path = os.path.join(UPLOAD_DIR, f"{video_id}.{ext}")
        if os.path.exists(potential_path):
            video_path = potential_path
            break
    if not video_path:
        logger.error(f"비디오 파일을 찾을 수 없습니다: video_id={video_id}")
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
    
    # 비디오 처리하여 피드백 생성
    feedback_data = process_video(video_path)
    
    # 피드백 데이터 확인
    if not feedback_data:
        logger.info(f"피드백 데이터가 없습니다: video_id={video_id}")
        # 'problem': 'none' 응답 추가
        return FeedbackResponse(
            feedbacks=[],
            message="피드백 데이터가 없습니다. 문제가 감지되지 않았습니다.",
            problem="none"
        )
    
    return FeedbackResponse(
        feedbacks=feedback_data,
        message="피드백 데이터 생성 완료",
        problem=None
    )
