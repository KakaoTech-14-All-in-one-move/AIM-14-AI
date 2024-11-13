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
# Docker 여부에 따라 경로 설정
try:
    if "docker" in open("/proc/1/cgroup").read():
        UPLOAD_DIR = Path("/tmp/storage/input_video")
        FEEDBACK_DIR = Path("/tmp/storage/output_feedback_frame")
    else:
        UPLOAD_DIR = Path("storage/input_video")
        FEEDBACK_DIR = Path("storage/output_feedback_frame")
except FileNotFoundError:
    # 로컬 경로로 설정
    UPLOAD_DIR = Path("storage/input_video")
    FEEDBACK_DIR = Path("storage/output_feedback_frame")

# 디렉터리가 없으면 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# 디렉터리가 없으면 생성
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
    if video_duration is None:
        logger.error(f"비디오 파일을 가져올 수 없습니다: {file_path}")
        return "video_error" # 비디오 파일 문제를 나타냄

    # 세그먼트 길이와 프레임 간격 설정
    segment_length = 60  # 초
    frame_interval = 3    # 초

    feedback_data = []

    # 디렉터리 경로가 지정되었는지 확인
    if FEEDBACK_DIR is None or not FEEDBACK_DIR.exists():
        logger.error(f"피드백 이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다: {FEEDBACK_DIR}")
        return "directory_error"

    # 각 세그먼트별로 프레임 추출 및 피드백 분석
    for start_time in range(0, int(video_duration), segment_length):
        segment_index = start_time // segment_length
        frames = download_and_sample_video_local(file_path, start_time, segment_length, frame_interval)

        if frames is None or len(frames) == 0:
            logger.warning(f"프레임을 추출할 수 없습니다. 비디오 파일에 문제가 있을 수 있습니다: {file_path}")
            return "codec_error"  # 코덱 문제로 인한 프레임 추출 실패를 명확히 나타냄

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
                    if not os.path.exists(image_path):
                        raise IOError("이미지가 지정된 경로에 저장되지 않았습니다.")
                except Exception as e:
                    logger.error(f"이미지 저장 중 오류 발생: {e}")
                    return "image_save_error"
        else:
            logger.warning(f"세그먼트 {segment_index+1}에서 프레임을 추출할 수 없습니다.")

    # 피드백 데이터 반환 (비어 있을 수 있음)        
    return feedback_data

@router.get("/video-send-feedback/{video_id}/", response_model=FeedbackResponse)
async def send_feedback_endpoint(video_id: str):
    """
    video_id를 통해 저장된 비디오 파일을 처리하고 피드백 데이터를 반환합니다.
    """
    # VP9 변환된 비디오 파일 경로 확인
    converted_video_filename = f"{video_id}_vp9.webm"
    video_path = os.path.join(UPLOAD_DIR, converted_video_filename)

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
        logger.error(f"비디오 파일을 찾을 수 없습니다: video_id={video_id}")
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
    
    # 비디오 처리하여 피드백 생성
    feedback_data = process_video(video_path)

    # 피드백 데이터 확인 - error_handling
    if feedback_data == "video_error": # 비디오 파일 자체에 문제 - 파일 손상 or 포맷 문제 (함수에서 비디오 길이를 확인할 때 발생)
        logger.info(f"비디오 길이 문제로 피드백 데이터를 생성하지 못했습니다: video_id={video_id}")
        return FeedbackResponse(
            feedbacks=[],
            message="비디오 길이 문제로 피드백 데이터를 생성하지 못했습니다.",
            problem="video_error"
        )

    # 비디오 파일은 정상 - 단, 특정 코텍 지원을 못해서 프레임 추출 불가
    elif feedback_data == "codec_error":
        logger.info(f"코덱 문제로 피드백 데이터를 생성하지 못했습니다: video_id={video_id}")
        return FeedbackResponse(
            feedbacks=[],
            message="코덱 문제로 피드백 데이터를 생성하지 못했습니다.",
            problem="codec_error"
        )
    
    # image_save_error: 이미지를 저장할 때 오류가 발생한 경우.
    elif feedback_data == "image_save_error": 
        logger.info(f"피드백 이미지 저장 오류로 피드백 데이터를 생성하지 못했습니다: video_id={video_id}")
        return FeedbackResponse(
            feedbacks=[],
            message="이미지 저장 오류로 피드백 데이터를 생성하지 못했습니다.",
            problem="image_save_error"
        )

    # directory_error: 피드백 이미지를 저장할 디렉터리가 없거나 지정되지 않은 경우.
    elif feedback_data == "directory_error":
        logger.info(f"이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다: video_id={video_id}")
        return FeedbackResponse(
            feedbacks=[],
            message="이미지를 저장할 디렉터리가 지정되지 않았거나 존재하지 않습니다.",
            problem="directory_error"
        )
    # no_feedback: 피드백할 내용이 없는 경우 - 피드백 내용이 none일때.
    elif not feedback_data: 
        logger.info(f"분석 결과 피드백할 내용이 없습니다: video_id={video_id}")
        return FeedbackResponse(
            feedbacks=[],
            message="분석 결과 피드백할 내용이 없습니다.",
            problem="no_feedback"
        )

    return FeedbackResponse(
        feedbacks=feedback_data,
        message="피드백 데이터 생성 완료",
        problem=None
    )
