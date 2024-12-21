# vlm_model/utils/analysis_video/load_prompt.py

from pathlib import Path
import logging
from fastapi import HTTPException
from vlm_model.config import PROMPT_PATH 
from vlm_model.exceptions import PromptImportingError

# 로거 설정
logger = logging.getLogger(__name__) 

def load_user_prompt() -> str:
    """
    프롬프트 파일을 로드합니다.
    """    
    try:
        with PROMPT_PATH.open('r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(f"프롬프트 파일을 찾을 수 없음: {PROMPT_PATH}", extra={
            "errorType": "FileNotFoundError",
            "error_message": f"프롬프트 파일을 찾을 수 없음: {PROMPT_PATH}"
        })
        raise PromptImportingError("프롬프트 파일을 찾을 수 없습니다") from e
    except Exception as e:
        logger.error(f"프롬프트 파일을 로드할 수 없음: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise PromptImportingError("프롬프트 파일을 로드하는 중 오류가 발생했습니다") from e
