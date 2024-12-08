# main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response

from vlm_model.exceptions import (
    VideoImportingError,
    PromptImportingError,
    VideoProcessingError,
    ImageEncodingError
)

from vlm_model.routers.upload_video import router as upload_video_router
from vlm_model.routers.send_feedback import router as send_feedback_router
from vlm_model.routers.delete_files import router as delete_files_router 

from pathlib import Path
import json
import uvicorn
import logging
import logging.config
import uuid
import traceback

# Context variables import
from vlm_model.context_var import request_id_ctx_var

# Middleware import
from vlm_model.middleware import RequestIDMiddleware

app = FastAPI()

# JSON 기반 로깅 설정 적용
logging_config_path = Path(__file__).resolve().parent / "logging_config.json"  # 프로젝트 루트에 위치한 파일 경로
with open(logging_config_path, "r") as f:
    logging_config = json.load(f)

logging.config.dictConfig(logging_config)
logger = logging.getLogger("vlm_model")


# 모든 출처를 허용하는 CORS 설정 (자격 증명 포함 불가)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # credentials를 반드시 False로 설정
)

# Request ID 미들웨어 추가
app.add_middleware(RequestIDMiddleware)

# 라우터 포함
app.include_router(upload_video_router, prefix="/api/video", tags=["Video Upload"])
app.include_router(send_feedback_router, prefix="/api/video", tags=["Feedback Retrieval"])
app.include_router(delete_files_router, prefix="/api/video", tags=["File Deletion"])

# 정적 파일을 제공할 디렉토리 설정 (선택 사항)
app.mount("/static", StaticFiles(directory="storage/output_feedback_frame"), name="static")

# 루트 엔드포인트
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello, Toby!"}

# 요청 로깅 미들웨어: 모든 요청과 응답을 로깅
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug("Request received")
    try:
        response = await call_next(request)
        # responseTime 계산: 요청 처리 시간
        # FastAPI 자체적으로 responseTime을 제공하지 않으므로, 직접 측정 필요
        # 여기서는 단순 예시로 timestamp 차이를 사용
        # 실제로는 start_time을 기록하고 response 후에 차이를 계산해야 함
        response_time = response.headers.get("X-Response-Time", "unknown")  # 필요 시 설정
        logger.info("Response sent", extra={
            "errorType": "",
            "error_message": ""
        })
        return response
    except Exception as e:
        # 예외 발생 위치 추출
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            filename, lineno, func, text = tb[-1]  # 가장 마지막 스택 프레임
        else:
            filename, lineno, func, text = "unknown", 0, "unknown", "unknown"
        
        logger.error("Error processing request", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise e


# 예외 처리 핸들러: VideoImportingError 발생 시
@app.exception_handler(VideoImportingError)
async def video_importing_exception_handler(request: Request, exc: VideoImportingError):
    """
    VideoImportingError 발생 시 400 Bad Request 응답을 반환합니다.
    """
    logger.error("Video importing error", extra={
        "errorType": "VideoImportingError",
        "error_message": exc.detail
    })
    return JSONResponse(
        status_code=400,  #  비디오 파일이 유효하지 않을 때 사용됩니다. 예를 들어, 잘못된 비디오 형식이나 손상된 파일을 업로드한 경우
        content={"detail": exc.detail},
    )

# 예외 처리 핸들러: PromptImportingError 발생 시
@app.exception_handler(PromptImportingError)
async def prompt_importing_exception_handler(request: Request, exc: PromptImportingError):
    """
    PromptImportingError 발생 시 400 Bad Request 응답을 반환합니다.
    """
    logger.error("PromptImportingError", extra={
        "errorType": "PromptImportingError",
        "error_message": exc.detail
    })
    return JSONResponse(
        status_code=400,  # 프롬프트 파일의 형식이 잘못되었거나, 필수 데이터가 누락된 경우
        content={"detail": exc.detail},
    )

# 예외 처리 핸들러: VideoProcessingError 발생 시
@app.exception_handler(VideoProcessingError)
async def video_processing_exception_handler(request: Request, exc: VideoProcessingError):
    """
    VideoProcessingError 발생 시 500 Internal Server Error 응답을 반환합니다.
    """
    logger.error("Video processing error", extra={
        "errorType": "VideoProcessingError",
        "error_message": exc.detail
    })
    return JSONResponse(
        status_code=500,  # 서버 내부에서 비디오를 처리하는 중 예상치 못한 오류가 발생했을 때 사용
        content={"detail": exc.detail},
    )

# 예외 처리 핸들러: ImageEncodingError 발생 시
@app.exception_handler(ImageEncodingError)
async def image_encoding_exception_handler(request: Request, exc: ImageEncodingError):
    """
    ImageEncodingError 발생 시 500 Internal Server Error 응답을 반환합니다.
    """
    logger.error("Image encoding error", extra={
        "errorType": "ImageEncodingError",
        "error_message": exc.detail
    })
    return JSONResponse(
        status_code=500,  # 이미지 인코딩 과정에서 서버 내부적인 문제가 발생했을 때 사용
        content={"detail": exc.detail},
    )

# 예외 처리 핸들러: 일반 예외 발생 시
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    예상치 못한 예외 발생 시 500 Internal Server Error 응답을 반환합니다.
    """
    logger.error("Unhandled exception", extra={
        "errorType": type(exc).__name__,
        "error_message": str(exc)
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."},
    )

# 테스트 엔드포인트 추가
@app.get("/test-logging")
def test_logging():
    logger.debug("디버그 레벨 로그 테스트")
    logger.info("정보 레벨 로그 테스트")
    logger.error("오류 레벨 로그 테스트")
    return {"message": "로깅 테스트 완료"}

# 서버 실행 (uvicorn.run()에서 log_config 지정)
if __name__ == "__main__":
    # 현재 작업 디렉터리를 스크립트의 디렉터리로 설정
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config="logging_config.json"  # 로깅 설정 파일 지정
    )

# 실행 명령어 (터미널에서 실행 시):
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-config logging_config.json