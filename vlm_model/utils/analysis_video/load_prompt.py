# vlm_model/utils/analysis/load_user_prompt.py

from pathlib import Path
import logging
from fastapi import HTTPException
from vlm_model.exceptions import PromptImportingError

# 로거 설정
logger = logging.getLogger(__name__) 

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
    except FileNotFoundError as e:
        logger.error(f"프롬프트 파일을 찾을 수 없음: {prompt_path}", extra={
            "errorType": "FileNotFoundError",
            "error_message": f"프롬프트 파일을 찾을 수 없음: {prompt_path}"
        })
        raise PromptImportingError("프롬프트 파일을 찾을 수 없습니다") from e
    except Exception as e:
        logger.error(f"프롬프트 파일을 로드할 수 없음: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise PromptImportingError("프롬프트 파일을 로드하는 중 오류가 발생했습니다") from e
