# utils/video_processing.py

import os
import re
import uuid
import base64
import logging
import openai
from pathlib import Path

from fastapi import HTTPException

from vlm_model.schemas.feedback import FeedbackFrame
from vlm_model.utils.download_video import download_and_sample_video_local
from vlm_model.utils.analysis import analyze_frames
from vlm_model.utils.analysis_video.load_prompt import load_user_prompt
from vlm_model.utils.analysis_video.parse_feedback import parse_feedback_text
from vlm_model.utils.encoding_image import encode_image
from vlm_model.utils.encoding_feedback_image import encode_feedback_image
from vlm_model.utils.video_duration import get_video_duration
from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import analyze_frame
from vlm_model.exceptions import VideoProcessingError, ImageEncodingError
from vlm_model.openai_config import SYSTEM_INSTRUCTION
from vlm_model.config import FEEDBACK_DIR

logger = logging.getLogger(__name__) 

def process_video(file_path: str, video_id: str):
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
    frame_interval = 1    # 초
    feedback_data = []
    mediapipe_results = []

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
            frames_low_res = download_and_sample_video_local(file_path, start_time, segment_length, frame_interval)
        except VideoProcessingError as vpe:
            logger.error(f"프레임을 추출할 수 없습니다: {vpe.message}", extra={
                "errorType": "VideoProcessingError",
                "error_message": f"프레임 추출 실패: {vpe.message}"
            })
            raise VideoProcessingError("프레임을 추출할 수 없습니다.") from vpe

        if frames_low_res is None or len(frames_low_res) == 0:
            logger.error(f"프레임을 추출할 수 없습니다. 비디오 파일에 문제가 있을 수 있습니다: {file_path}", extra={
                "errorType": "VideoProcessingError",
                "error_message": f"비디오 파일에 문제가 있을 수 있습니다. {file_path}"
            })
            raise VideoProcessingError("비디오 파일에 문제가 있을 수 있습니다.")

        # Mediapipe 기반 문제 프레임 필터링
        problematic_frames = []
        mediapipe_results_segment = []  # 세그먼트별 Mediapipe 결과 저장
        problematic_timestamps = []  # 타임스탬프 리스트 추가
        previous_pose_landmarks = None
        previous_hand_landmarks = None

        for idx, frame_low_res in enumerate(frames_low_res):
            # 기본값으로 초기화
            mediapipe_feedback = {
                "posture_score": 0.0,
                "gaze_score": 0.0,
                "gestures_score": 0.0,
                "sudden_movement_score": 0.0
            }
            current_pose_landmarks = previous_pose_landmarks
            current_hand_landmarks = previous_hand_landmarks

            # analyze_frame 호출 및 결과 처리
            try:
                mediapipe_feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(
                    frame_low_res, previous_pose_landmarks, previous_hand_landmarks
                )
            except Exception as e:
                logger.error(f"프레임 {idx} 분석 중 오류 발생: {str(e)}", extra={
                    "errorType": type(e).__name__,
                    "error_message": str(e)
                })
                continue  # 다음 프레임으로 계속 진행

            # 특정 기준을 초과하는 경우 문제 프레임으로 간주
            if (mediapipe_feedback["posture_score"] > 0.8 or
                mediapipe_feedback["gaze_score"] > 0.7 or
                mediapipe_feedback["gestures_score"] > 0.7 or
                mediapipe_feedback["sudden_movement_score"] > 0.7):

                # 문제 프레임 및 Mediapipe 결과 저장
                timestamp_sec = start_time + idx * frame_interval  # 타임스탬프 계산
                problematic_frames.append((frame_low_res, segment_index, idx, timestamp_sec))
                problematic_timestamps.append(timestamp_sec)  # 타임스탬프 추가

                mediapipe_results_segment.append({
                    "gaze_processing": {
                        "score": mediapipe_feedback["gaze_score"]
                    },
                    "gestures": {
                        "score": mediapipe_feedback["gestures_score"]
                    },
                    "posture_body": {
                        "score": mediapipe_feedback["posture_score"]
                    },
                    "movement": {
                        "score": mediapipe_feedback["sudden_movement_score"]
                    }
                })

            # 이전 랜드마크 갱신
            previous_pose_landmarks = current_pose_landmarks if current_pose_landmarks else previous_pose_landmarks
            previous_hand_landmarks = current_hand_landmarks if current_hand_landmarks else previous_hand_landmarks

        # 문제가 되는 프레임만 처리
        if problematic_frames:
            try:
                frames_to_analyze = [frame_info[0] for frame_info in problematic_frames]
                timestamps_to_analyze = [frame_info[3] for frame_info in problematic_frames]  # 초 단위 타임스탬프 전달
                mediapipe_results_subset = mediapipe_results_segment[:len(problematic_frames)]

                problematic_frames_processed, feedbacks = analyze_frames(
                    frames=frames_to_analyze,
                    timestamps=timestamps_to_analyze,  # 초 단위 타임스탬프 전달
                    mediapipe_results=mediapipe_results_subset,
                    segment_idx=segment_index,
                    duration=segment_length,
                    segment_length=segment_length,
                    system_instruction=SYSTEM_INSTRUCTION,
                    frame_interval=frame_interval
                )
            except Exception as e:
                logger.error(f"프레임 분석 중 오류 발생: {str(e)}",  extra={
                    "errorType": type(e).__name__,
                    "error_message": str(e)
                })
                raise HTTPException(status_code=422, detail="프레임 분석 중 오류가 발생했습니다.") from e
        else:
            problematic_frames_processed = []
            feedbacks = []

        logger.debug(f"프레임 수: {len(problematic_frames_processed)}, 피드백 수: {len(feedbacks)}")

        for frame_info, feedback_text in zip(problematic_frames_processed, feedbacks):
            frame_low_res, segment_number, frame_number, timestamp = frame_info  # timestamp는 float

            # 이미지 인코딩 (Base64)
            try:
                image_base64 = encode_feedback_image(frame_low_res)
                if not image_base64:
                    logger.error(f"프레임 {frame_number}의 이미지 인코딩 실패", extra={
                        "errorType": "ImageEncodingError",
                        "error_message": f"이미지 인코딩 실패. 프레임 {frame_number}"
                    })
                    raise ImageEncodingError("이미지 인코딩이 실패했습니다.")
            except ImageEncodingError as iee:
                logger.error(f"이미지 인코딩 실패: {iee}", extra={
                    "errorType": "ImageEncodingError",
                    "error_message": f"이미지 인코딩 실패: {iee}"
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

            # 초 단위 타임스탬프를 "Xm Ys" 형식으로 변환
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes}m {seconds}s"

            # FeedbackFrame creation:
            feedback_frame = FeedbackFrame(
                video_id=video_id,
                frame_index=frame_number,
                timestamp=timestamp_str,  # 문자열 타임스탬프 전달
                feedback_text=feedback_sections,
                image_base64=image_base64
            )
            feedback_data.append(feedback_frame.dict())

            # 피드백 이미지를 저장하는 경우
            if FEEDBACK_DIR:
                # 초 단위 타임스탬프를 "Xm Ys" 형식으로 변환 (이미 변환됨)
                # safe_timestamp는 timestamp_str을 기반으로 생성
                safe_timestamp = re.sub(r'[^\w_]', '', timestamp_str.replace("m ", "m_").replace(" ", "_").replace("s", "s_").strip("_"))
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
