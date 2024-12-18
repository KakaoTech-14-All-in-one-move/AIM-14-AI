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

def convert_to_vp9(input_path: str, output_path: str, preset: str = 'faster', cpu_used: int = 8, threads: int = 0, tile_columns: int = 4, tile_rows: int = 2, bitrate: str = '1M') -> bool:
    """
    H.264 코덱 비디오를 VP9 코덱으로 변환합니다.

    Parameters:
    - input_path: 입력 비디오 파일 경로
    - output_path: 출력 비디오 파일 경로
    - preset: 인코딩 프리셋 (예: 'faster', 'fast', 'medium', 'slow', 'best')
    - cpu_used: 인코딩 속도 조절 (0-8, 높을수록 빠름)
    - threads: FFmpeg가 사용할 스레드 수 (0은 자동)
    - tile_columns: 타일 열 수
    - tile_rows: 타일 행 수
    - bitrate: 비디오 비트레이트 (예: '1M', '800k')

    Returns:
    - bool: 변환이 성공하면 True, 그렇지 않으면 예외를 발생시킵니다.
    """
    try:
        command = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libvpx-vp9',
            '-b:v', bitrate,
            '-preset', preset,
            '-cpu-used', str(cpu_used),
            '-c:a', 'libopus',
            '-threads', str(threads),
            '-tile-columns', str(tile_columns),
            '-tile-rows', str(tile_rows),
            output_path
        ]

        logger.debug(f"FFmpeg 명령어: {' '.join(command)}")
        subprocess.run(command, check=True)
        logger.info(f"비디오 변환 성공: {output_path}")
        return True

        logger.debug(f"FFmpeg 명령어: {' '.join(command)}")
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

def get_video_codec_info(video_path: str) -> str:
    """
    FFmpeg로 비디오 파일의 코덱 정보를 확인합니다.
    """
    try:
        command = ['ffmpeg', '-i', video_path]
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = result.stderr.decode()
        logger.debug(f"코덱 정보: {output}")

        # 코덱 정보를 추출하기 위한 정규 표현식
        codec_match = re.search(r'Stream #\d+:\d+.*Video: ([^ ,]+)', output)
        if codec_match:
            codec_info = codec_match.group(1)
            logger.debug(f"추출된 코덱 정보: {codec_info}")
            return codec_info
        else:
            logger.error(f"비디오 코덱 정보를 찾을 수 없습니다: {video_path}")
            return ""

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

def is_vp9(video_path: str) -> bool:
    """
    주어진 비디오 파일이 VP9 코덱인지 확인합니다.
    
    Parameters:
    - video_path: 비디오 파일 경로
    
    Returns:
    - bool: VP9 코덱인 경우 True, 아니면 False
    """
    codec_info = get_video_codec_info(video_path)
    is_vp9_codec = 'vp9' in codec_info.lower()
    logger.debug(f"파일 {video_path}의 VP9 여부: {is_vp9_codec}")
    return is_vp9_codec

def convert_to_vp9_if_needed(input_path: str, output_path: str, preset: str = 'faster', cpu_used: int = 8, threads: int = 0, tile_columns: int = 4, tile_rows: int = 2, bitrate: str = '1M') -> bool:
    """
    주어진 비디오 파일이 VP9 코덱이 아닌 경우에만 VP9으로 변환합니다.
    
    Parameters:
    - input_path: 입력 비디오 파일 경로
    - output_path: 출력 비디오 파일 경로
    - preset: 인코딩 프리셋 (예: 'faster', 'fast', 'medium', 'slow', 'best')
    - cpu_used: 인코딩 속도 조절 (0-8, 높을수록 빠름)
    - threads: FFmpeg가 사용할 스레드 수 (0은 자동)
    - tile_columns: 타일 열 수
    - tile_rows: 타일 행 수
    - bitrate: 비디오 비트레이트 (예: '1M', '800k')
    
    Returns:
    - bool: 변환이 수행되면 True, 이미 VP9인 경우 False
    """
    if is_vp9(input_path):
        logger.info(f"이미 VP9 코덱인 파일입니다: {input_path}")
        return False
    return convert_to_vp9(
        input_path=input_path,
        output_path=output_path,
        preset=preset,
        cpu_used=cpu_used,
        threads=threads,
        tile_columns=tile_columns,
        tile_rows=tile_rows,
        bitrate=bitrate
    )
