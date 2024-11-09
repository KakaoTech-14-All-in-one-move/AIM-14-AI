# utils/analysis.py

import re
from typing import List, Tuple
import openai
import numpy as np
import math
from vlm_model.config import SYSTEM_INSTRUCTION
from vlm_model.constants.behaviors import PROBLEMATIC_BEHAVIORS
from vlm_model.utils.encoding_image import encode_image

# OpenAI 모듈을 client로 정의
client = openai

def analyze_frames(frames: List[np.ndarray], segment_idx: int, duration: int, segment_length: int, system_instruction: str, frame_interval: int = 3) -> Tuple[List[Tuple[np.ndarray, int, int, str]], List[str]]:
    problematic_frames = []
    feedbacks = []

    num_frames = len(frames)
    time_stamps = [
        segment_idx * segment_length + i * frame_interval
        for i in range(num_frames)
    ]

    for i, (frame, frame_time_sec) in enumerate(zip(frames, time_stamps)):
        minutes = int(frame_time_sec // 60)
        seconds = int(frame_time_sec % 60)
        timestamp = f"{minutes}m {seconds}s"

        # 사용자 프롬프트 생성
        user_prompt = (
            "다음 이미지에서 발표자의 비언어적 행동을 분석하고, system_instruction 내용에 기반해서 문제가 되는 행동이 있으면 피드백을 제공해주세요. "
            "문제가 없으면 '문제 없음'이라고 답해주세요."
        )

        img_type = "image/jpeg"

        # 이미지를 인코딩
        img_b64_str = encode_image(frame)

        if img_b64_str is None:
            continue

        # 사용자 메시지 구성
        user_message = f"{user_prompt}\n\n이미지 데이터: data:{img_type};base64,{img_b64_str}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 사용 가능한 모델 이름으로 설정
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
                max_tokens=600,
            )

            # 생성된 텍스트과 문제 행동 추출
            generated_text = response.choices[0].message.content
            behaviors_detected = re.findall(r'\[([^\[\]]+)\]', generated_text)

            # 공백 제거
            behaviors_detected = [behavior.strip() for behavior in behaviors_detected]

            # 디버깅을 위해 감지된 문제 행동 출력
            print(f"[디버그] 프레임 {i+1} 응답 텍스트: {generated_text}")
            print(f"[디버그] 감지된 문제 행동: {behaviors_detected}")
            print(f"[디버그] PROBLEMATIC_BEHAVIORS 리스트: {PROBLEMATIC_BEHAVIORS}")

            # 문제 행동 감지 여부 확인
            problem_detected = any(behavior in PROBLEMATIC_BEHAVIORS for behavior in behaviors_detected)

            if problem_detected:
                # 프레임과 세그먼트 정보를 저장
                problematic_frames.append((frame, segment_idx + 1, i + 1, timestamp))
                feedbacks.append(generated_text)

        except client.error.OpenAIError as e:
            print(f"프레임 {i+1} 처리 중 OpenAI 오류 발생: {e}")
        except Exception as e:
            print(f"프레임 {i+1} 처리 중 오류 발생: {e}")

    return problematic_frames, feedbacks
