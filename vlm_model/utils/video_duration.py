# utils/video_duration.py

import subprocess
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_video_duration(video_path: str) -> Optional[float]:
    """
    ffmpeg를 사용하여 비디오 길이를 초 단위로 반환합니다.
    """
    try:
        logger.info(f"ffprobe로 비디오 길이를 가져오는 중: {video_path}")
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        logger.info(f"비디오 길이: {duration}초")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe 실행 오류: {e.stderr}")
    except (KeyError, ValueError) as e:
        logger.error(f"비디오 길이 파싱 오류: {e}")
    except Exception as e:
        logger.exception(f"비디오 길이를 가져오는 중 예기치 못한 오류 발생: {e}")
    return None

