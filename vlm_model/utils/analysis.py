# utils/analysis.py

import json
import re
from typing import List, Tuple
from openai import (
    AuthenticationError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
    OpenAIError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError
)
from openai import OpenAI # import openai
import numpy as np
from pathlib import Path
import logging
from fastapi import HTTPException

from vlm_model.openai_config import SYSTEM_INSTRUCTION
from vlm_model.utils.encoding_image import encode_image
from vlm_model.utils.analysis_video.load_prompt import load_user_prompt
from vlm_model.utils.analysis_video.parse_feedback import parse_feedback_text
from vlm_model.schemas.feedback import FeedbackSections, FeedbackDetails
from vlm_model.exceptions import PromptImportingError

# 모듈별 로거 생성
logger = logging.getLogger(__name__) 

# OpenAI 모듈을 client로 정의
client = OpenAI() # openai

def analyze_frames(frames: List[np.ndarray], timestamps: List[float], mediapipe_results: List[dict], segment_idx: int, duration: int, segment_length: int, system_instruction: str, frame_interval: int = 1) -> Tuple[List[Tuple[np.ndarray, int, int, str]], List[str]]:
    """
    주어진 프레임들을 분석하여 문제 행동을 감지하고 피드백을 생성합니다.

    Parameters:
    - frames: 분석할 프레임들의 NumPy 배열
    - segment_idx: 현재 세그먼트의 인덱스
    - duration: 세그먼트의 지속 시간 (초 단위)
    - segment_length: 세그먼트의 길이 (초 단위)
    - system_instruction: 시스템 지침 문자열
    - frame_interval: 프레임 추출 간격 (초 단위)

    Returns:
    - problematic_frames: 문제 행동이 감지된 프레임 정보 리스트
    - feedbacks: 생성된 피드백 리스트
    """
    if not mediapipe_results:
        logger.error(f"Mediapipe 결과가 비어 있습니다. mediapipe_results: {mediapipe_results}")
        raise ValueError("Mediapipe 결과가 비어 있습니다. 필수 입력값입니다.")

    problematic_frames = []
    feedbacks = []

    num_frames = len(frames)

    # 프롬프트 파일 절대 경로 설정
    user_prompt = load_user_prompt()

    # mediapipe_results의 길이와 frames의 길이가 동일한지 확인
    if len(mediapipe_results) != len(frames):
        logger.error("mediapipe_results의 길이와 frames의 길이가 일치하지 않습니다.")
        raise ValueError("mediapipe_results의 길이와 frames의 길이가 일치하지 않습니다.")

    for i, (frame, timestamp, mediapipe_feedback) in enumerate(zip(frames, timestamps, mediapipe_results)):
        try:
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes}m {seconds}s"
        except Exception as e:
            logger.error(f"타임스탬프 변환 중 오류 발생: {e}", extra={
                "errorType": "TimestampConversionError",
                "error_message": str(e)
            })
            timestamp_str = "0m 0s"  # 기본값 설정 또는 적절한 처리

        img_type = "image/jpeg"

        # Mediapipe에서 필터링된 결과를 메시지에 포함
        mediapipe_feedback_text = "\n".join(
            [f"{key}: {value}" for key, value in mediapipe_feedback.items()]
        )

        # 이미지를 인코딩
        img_b64_str = encode_image(frame)

        if img_b64_str is None:
            continue

        # 사용자 메시지 구성
        user_message = f"{user_prompt}\n\nMediapipe에서 감지된 문제 행동:\n{mediapipe_feedback_text}\n\n이미지 데이터: data:{img_type};base64,{img_b64_str}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                max_tokens=2000,
                temperature=0.4,
                top_p=0.3
            )
            logger.info(f"프레임 {i+1} OpenAI 응답: {response}")

            # 생성된 텍스트과 문제 행동 추출
            generated_text = response.choices[0].message.content

            # 코드 블록 제거 (```json\n ... \n```)
            clean_text = re.sub(r'^```json\s*', '', generated_text, flags=re.MULTILINE)
            clean_text = re.sub(r'```\s*$', '', clean_text, flags=re.MULTILINE)

            # JSON 형식으로 응답을 파싱
            feedback_sections = parse_feedback_text(clean_text)

            # 문제 행동 감지 여부 확인
            problem_detected = any(
                getattr(feedback_sections, field).improvement for field in feedback_sections.__fields__
            )

            # 디버깅을 위해 감지된 문제 행동 출력
            detected_behaviors = [
                field for field in feedback_sections.__fields__ 
                if getattr(feedback_sections, field).improvement
            ]
            logger.debug(f"프레임 {i+1} 응답 텍스트: {generated_text}")
            logger.debug(f"감지된 문제 행동: {detected_behaviors}")

            if problem_detected:
                # 프레임과 세그먼트 정보를 저장
                problematic_frames.append((frame, segment_idx + 1, i + 1, timestamp))
                feedbacks.append(generated_text)

        except AuthenticationError as e:
            # 401 - Invalid Authentication
            logger.error(f"프레임 {i+1} 처리 중 인증 오류 발생: {e}", extra={
                "errorType": "AuthenticationError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=401, detail="인증 오류: API 키를 확인해주세요.") from e

        except PermissionDeniedError as e:
            # 403 - Permission Denied (e.g., Country not supported)
            logger.error(f"프레임 {i+1} 처리 중 권한 오류 발생: {e}", extra={
                "errorType": "PermissionDeniedError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=403, detail="권한 오류: API 사용 권한을 확인해주세요.") from e

        except RateLimitError as e:
            # 429 - Rate Limit Exceeded
            logger.error(f"프레임 {i+1} 처리 중 Rate Limit 초과: {e}", extra={
                "errorType": "RateLimitError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=429, detail="요청 제한 초과: 요청 속도를 줄여주세요.") from e

        except BadRequestError as e:
            # 400 - Bad Request
            logger.error(f"프레임 {i+1} 처리 중 잘못된 요청 오류 발생: {e}", extra={
                "errorType": "BadRequestError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=400, detail="잘못된 요청: 요청 데이터를 확인해주세요.") from e

        except ConflictError as e:
            # 409 - Conflict
            logger.error(f"프레임 {i+1} 처리 중 충돌 오류 발생: {e}", extra={
                "errorType": "ConflictError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=409, detail="충돌 오류: 요청을 다시 시도해주세요.") from e

        except InternalServerError as e:
            # 500 - Internal Server Error
            logger.error(f"프레임 {i+1} 처리 중 내부 서버 오류 발생: {e}", extra={
                "errorType": "InternalServerError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=502, detail="내부 서버 오류: 나중에 다시 시도해주세요.") from e

        except NotFoundError as e:
            # 404 - Not Found
            logger.error(f"프레임 {i+1} 처리 중 자원 미존재 오류 발생: {e}", extra={
                "errorType": "NotFoundError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=404, detail="자원이 존재하지 않습니다.") from e

        except UnprocessableEntityError as e:
            # 422 - Unprocessable Entity
            logger.error(f"프레임 {i+1} 처리 중 처리 불가능한 엔티티 오류 발생: {e}", extra={
                "errorType": "UnprocessableEntityError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=422, detail="처리 불가능한 데이터입니다.") from e

        except APIError as e:
            # 502 - Bad Gateway
            logger.error(f"프레임 {i+1} 처리 중 API 오류 발생: {e}", extra={
                "errorType": "APIError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=502, detail="서버 오류: 나중에 다시 시도해주세요.") from e

        except APITimeoutError as e:
            # 504 - Gateway Timeout
            logger.error(f"프레임 {i+1} 처리 중 API 타임아웃 오류 발생: {e}", extra={
                "errorType": "APITimeoutError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=504, detail="서버 응답 지연: 나중에 다시 시도해주세요.") from e

        except APIConnectionError as e:
            # 503 - Service Unavailable
            logger.error(f"프레임 {i+1} 처리 중 API 연결 오류 발생: {e}", extra={
                "errorType": "APIConnectionError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=503, detail="연결 오류: 네트워크 상태를 확인해주세요.") from e

        except OpenAIError as e:
            # 500 - OpenAI 관련 기타 오류
            logger.error(f"프레임 {i+1} 처리 중 OpenAI 라이브러리 오류 발생: {e}", extra={
                "errorType": "OpenAIError",
                "error_message": str(e)
            })
            raise HTTPException(status_code=500, detail="OpenAI 처리 중 알 수 없는 오류가 발생했습니다.") from e

        except ValueError as ve:
            # JSON 디코딩 오류 등
            logger.error(f"프레임 {i+1} 피드백 파싱 중 오류 발생: {ve}", extra={
                "errorType": "ValueError",
                "error_message": str(ve)
            })
            raise HTTPException(status_code=400, detail="피드백 파싱 과정 중 오류.") from ve

        except Exception as e:
            # 기타 모든 예외
            logger.error(f"프레임 {i+1} 처리 중 예기치 않은 오류 발생: {e}", extra={
                "errorType": type(e).__name__,
                "error_message": str(e)
            })
            raise HTTPException(status_code=500, detail="프레임 처리 중 예기치 않은 오류가 발생했습니다.") from e

    return problematic_frames, feedbacks
