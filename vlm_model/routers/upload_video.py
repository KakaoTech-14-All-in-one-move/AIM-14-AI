# vlm_model/routers/upload_video.py

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Response
import os
import uuid
import logging
import shutil
import subprocess
from pathlib import Path

from vlm_model.schemas.feedback import UploadResponse
from vlm_model.utils.video_duration import get_video_duration
from vlm_model.config import UPLOAD_DIR
from vlm_model.exceptions import VideoImportingError

router = APIRouter()

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
        }, exc_info=True)
        raise VideoImportingError("ffmpeg 설치가 필요합니다.") from f
        
    except subprocess.CalledProcessError as e:
        logger.error(f"비디오 변환 실패: {e.stderr.decode().strip()}", extra={
            "errorType": "CalledProcessError",
            "error_message": str(e)
        }, exc_info=True)
        raise VideoImportingError("비디오 변환 중 오류가 발생했습니다.") from e

    except Exception as e:
        logger.error(f"알 수 없는 변환 오류 발생: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        raise VideoImportingError("예기치 않은 변환 오류가 발생했습니다.") from e


@router.post("/receive-video/", response_model=UploadResponse)
async def receive_video_endpoint(response: Response, file: UploadFile = File(...)):
    """
    비디오 파일을 업로드 받아 저장하고, video_id를 반환합니다.
    """

    # 응답 헤더에 CORS 설정 추가
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "False"

    # 요청 수신 로그
    logger.info("receive_video_endpoint 호출됨")
    logger.info(f"받은 파일: {file.filename}")

    # 지원하는 파일 형식 확인
    ALLOWED_EXTENSIONS = {"webm", "mp4", "mov", "avi", "mkv"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.error(f"지원하지 않는 파일 형식: {file_extension}", extra={
            "errorType": "UnsupportedMediaType",
            "error_message": f"지원하지 않는 파일 형식 입니다.: {file_extension}"
        }, exc_info=True)
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 고유한 video_id 생성
    video_id = uuid.uuid4().hex

    # 비디오 파일 저장 경로 설정
    original_file_path = UPLOAD_DIR / f"{video_id}_original.{file_extension}"
    converted_file_path = UPLOAD_DIR / f"{video_id}_vp9.webm"

    # 비디오 파일 저장
    try:
        with open(original_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 파일 존재 여부와 크기 확인
        if not os.path.exists(original_file_path):
            logger.error("파일이 저장되지 않았습니다.", extra={
                "errorType": "VideoImportingError",
                "error_message": "파일이 저장되지 않았습니다."
            }, exc_info=True)
            raise VideoImportingError("파일이 저장되지 않았습니다.")

        file_size = os.path.getsize(original_file_path)
        logger.info(f"파일이 성공적으로 저장되었습니다. 크기: {file_size} bytes")

        # VP9 변환
        if convert_to_vp9(original_file_path, converted_file_path):
            return UploadResponse(
                video_id=video_id,
                message=f"비디오 업로드 및 VP9 변환 완료. 피드백 데이터를 받으려면 /send-feedback/{video_id} 엔드포인트를 호출하세요."
            )
        else:
            # 변환 실패 시 원본 파일 삭제
            os.remove(original_file_path)
            logger.error("비디오 코덱 변환이 실패했습니다.", extra={
                "errorType": "VideoImportingError",
                "error_message": "비디오 코덱 변환이 실패했습니다."
            }, exc_info=True)
            raise VideoImportingError("비디오 코덱 변환이 실패했습니다.")

    except VideoImportingError as vie:
        # 전역 예외 핸들러가 이를 처리
        raise vie

    except IOError as e:
        logger.error(f"파일 저장 중 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        raise VideoImportingError("파일 저장 중 오류가 발생했습니다.") from e

    except Exception as e:
        logger.error(f"알 수 없는 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        raise VideoImportingError("파일 처리 중 예기치 않은 오류가 발생했습니다.") from e

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
        }, exc_info=True)
        raise VideoImportingError("코덱 정보 확인 실패") from e

    except Exception as e:
        logger.error(f"코덱 정보 확인 중 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        raise VideoImportingError("코덱 정보 확인 중 오류 발생") from e