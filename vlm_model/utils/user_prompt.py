# utils/user_prompt.py

def generate_user_prompt(img_type: str, img_b64_str: str) -> str:
    """
    주어진 이미지 데이터와 타입을 기반으로 비언어적 행동 평가를 요청하는 사용자 프롬프트를 생성합니다.
    """
    return f"Please evaluate the presenter's non-verbal behavior.\n\nImage data: data:{img_type};base64,{img_b64_str}"