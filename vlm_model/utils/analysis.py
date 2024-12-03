# utils/analysis.py

import json
from typing import List, Tuple
from openai import OpenAI # import openai
import numpy as np
from vlm_model.config import SYSTEM_INSTRUCTION
from vlm_model.constants.behaviors import PROBLEMATIC_BEHAVIORS
from vlm_model.utils.encoding_image import encode_image
from vlm_model.schemas.feedback import FeedbackSections, FeedbackDetails
from pathlib import Path
from vlm_model.utils.retrieve_feedback import retrieve_relevant_feedback

# OpenAI 모듈을 client로 정의
client = OpenAI() # openai

def load_user_prompt() -> str:
    """
    프롬프트 파일을 로드합니다.
    """
    # 현재 파일의 위치를 기준으로 상대 경로 설정
    current_dir = Path(__file__).parent
    prompt_path = current_dir.parent.parent / 'prompt.txt'  # 실제 프로젝트 구조에 맞게 조정
    
    try:
        with prompt_path.open('r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    except Exception as e:
        raise Exception(f"프롬프트 파일을 로드하는 중 오류 발생: {e}")

def parse_feedback_text(feedback_text: str) -> FeedbackSections:
    """
    feedback_text를 FeedbackSections 형식으로 파싱합니다.
    """
    try:
        feedback_json = json.loads(feedback_text)
        if feedback_json.get("problem") == "none":
            return FeedbackSections(
                gaze_processing=FeedbackDetails(improvement="", recommendations=""),
                facial_expression=FeedbackDetails(improvement="", recommendations=""),
                gestures=FeedbackDetails(improvement="", recommendations=""),
                posture_body=FeedbackDetails(improvement="", recommendations=""),
                movement=FeedbackDetails(improvement="", recommendations="")
            )
        
        # 섹션별로 데이터 추출
        feedback_data = {}
        for section_key, field_name in {
            "gaze_processing": "gaze_processing",
            "facial_expression": "facial_expression",
            "gestures": "gestures",
            "posture_body": "posture_body",
            "movement": "movement"
        }.items():
            section = feedback_json.get(section_key, {})
            improvement = section.get("improvement", "").strip()
            recommendations = section.get("recommendations", "").strip()
            feedback_data[field_name] = FeedbackDetails(
                improvement=improvement,
                recommendations=recommendations
            )
        
        return FeedbackSections(**feedback_data)
    
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 디코딩 오류: {str(e)}")
    except Exception as e:
        raise Exception(f"FeedbackSections 생성 실패: {str(e)}")

def analyze_frames(frames: List[np.ndarray], segment_idx: int, duration: int, segment_length: int, system_instruction: str, model, index, documents, frame_interval: int = 3) -> Tuple[List[Tuple[np.ndarray, int, int, str]], List[str]]:
    """
    주어진 프레임들을 분석하여 문제 행동을 감지하고 피드백을 생성합니다.

    Parameters:
    - frames: 분석할 프레임들의 NumPy 배열
    - segment_idx: 현재 세그먼트의 인덱스
    - duration: 세그먼트의 지속 시간 (초 단위)
    - segment_length: 세그먼트의 길이 (초 단위)
    - system_instruction: 시스템 지침 문자열
    - model: SentenceTransformer 모델
    - index: FAISS 인덱스
    - documents: 문서 리스트
    - frame_interval: 프레임 추출 간격 (초 단위)

    Returns:
    - problematic_frames: 문제 행동이 감지된 프레임 정보 리스트
    - feedbacks: 생성된 피드백 리스트
    """
    problematic_frames = []
    feedbacks = []

    num_frames = len(frames)
    time_stamps = [
        segment_idx * segment_length + i * frame_interval
        for i in range(num_frames)
    ]

    # 프롬프트 파일 절대 경로 설정
    user_prompt = load_user_prompt()

    for i, (frame, frame_time_sec) in enumerate(zip(frames, time_stamps)):
        minutes = int(frame_time_sec // 60)
        seconds = int(frame_time_sec % 60)
        timestamp = f"{minutes}m {seconds}s"

        img_type = "image/jpeg"

        # 이미지를 인코딩
        img_b64_str = encode_image(frame)

        if img_b64_str is None:
            continue

        # 사용자 메시지 구성
        user_message = f"{user_prompt}\n\n이미지 데이터: data:{img_type};base64,{img_b64_str}"

        # RAG를 사용하여 관련 피드백 검색
        retrieved_context = retrieve_relevant_feedback(user_message, model, index, documents, top_k=3)
        retrieved_context_text = "\n".join(retrieved_context)

        # GPT API를 사용하여 피드백 생성
        prompt = (
            f"{system_instruction}\n\n"
            f"관련 정보:\n{retrieved_context_text}\n\n"
            f"{user_message}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 올바른 모델 이름으로 수정
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
            )

            # 생성된 텍스트과 문제 행동 추출
            generated_text = response.choices[0].message.content

            # JSON 형식으로 응답을 파싱
            feedback_sections = parse_feedback_text(generated_text)

            # 문제 행동 감지 여부 확인
            problem_detected = any(
                getattr(feedback_sections, field).improvement for field in feedback_sections.__fields__
            )

            # 디버깅을 위해 감지된 문제 행동 출력
            detected_behaviors = [
                field for field in feedback_sections.__fields__ 
                if getattr(feedback_sections, field).improvement
            ]
            print(f"[디버그] 프레임 {i+1} 응답 텍스트: {generated_text}")
            print(f"[디버그] 감지된 문제 행동: {detected_behaviors}")
            print(f"[디버그] PROBLEMATIC_BEHAVIORS 리스트: {PROBLEMATIC_BEHAVIORS}")

            if problem_detected:
                # 프레임과 세그먼트 정보를 저장
                problematic_frames.append((frame, segment_idx + 1, i + 1, timestamp))
                feedbacks.append(generated_text)

        except client.error.OpenAIError as e:
            print(f"프레임 {i+1} 처리 중 OpenAI 오류 발생: {e}")
        except ValueError as ve:
            print(f"프레임 {i+1} 피드백 파싱 중 오류 발생: {ve}")
        except Exception as e:
            print(f"프레임 {i+1} 처리 중 오류 발생: {e}")

    return problematic_frames, feedbacks
