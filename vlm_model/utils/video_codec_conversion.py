# vlm_model/utils/video_codec_conversion.py

import os
import re
import uuid
import base64
import logging
import subprocess
from pathlib import Path

from fastapi import HTTPException

from vlm_model.exceptions import VideoImportingError

logger = logging.getLogger(__name__) # 로거 사용

def convert_to_vp9(input_path: str, output_path: str) -> bool:
    """
    H.264 코덱 비디오를 VP9 코덱으로 변환합니다.
    """
    try:
        command = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libvpx-vp9', '-b:v', '1M',
            '-c:a', 'libopus', output_path
        ]
        subprocess.run(command, check=True)
        logger.info(f"비디오 변환 성공: {output_path}")
        return True
    
    except FileNotFoundError as f:
        logger.error(f"ffmpeg 패키지 파일을 찾을 수 없습니다. Dockerfile에 ffmpeg 설치를 추가했는지 확인하세요.", extra={
            "errorType": "FileNotFoundError",
            "error_message": str(f)
        })
        raise HTTPException(status_code=404, detail="ffmpeg 패키지 파일을 찾을수 없습니다. 설치가 필요합니다.") from f
        
    except subprocess.CalledProcessError as e:
        logger.error(f"비디오 변환 실패: {e.stderr.decode().strip()}", extra={
            "errorType": "CalledProcessError",
            "error_message": str(e)
        })
        raise VideoImportingError("비디오 변환 중 오류가 발생했습니다.") from e

    except Exception as e:
        logger.error(f"알 수 없는 변환 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise VideoImportingError("예기치 않은 변환 오류가 발생했습니다.") from e

def get_video_codec_info(video_path: str):
    """
    FFmpeg로 비디오 파일의 코덱 정보를 확인합니다.
    """
    try:
        command = ['ffmpeg', '-i', video_path]
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = result.stderr.decode()
        logger.debug(f"코덱 정보: {output}")

    except subprocess.CalledProcessError as e:
        logger.error(f"코덱 정보 확인 실패: {str(e)}", extra={
            "errorType": "CalledProcessError",
            "error_message": str(e)
        })
        raise VideoImportingError("코덱 정보 확인 실패") from e

    except Exception as e:
        logger.error(f"코덱 정보 확인 중 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise VideoImportingError("코덱 정보 확인 중 오류 발생") from e