# main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response

from vlm_model.routers.upload_video import router as upload_video_router
from vlm_model.routers.send_feedback import router as send_feedback_router

import uvicorn
import logging

# 로깅 설정
logger = logging.getLogger("uvicorn")

app = FastAPI()

# 모든 출처를 허용하는 CORS 설정 (자격 증명 포함 불가)
# CORS 미들웨어 설정 - 모든 출처 허용, credentials는 False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=False,  # credentials를 반드시 False로 설정
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 라우터 포함
app.include_router(upload_video_router, prefix="/api/video", tags=["Video Upload"])
app.include_router(send_feedback_router, prefix="/api/video", tags=["Feedback Retrieval"])

# 정적 파일을 제공할 디렉토리 설정 (선택 사항)
app.mount("/static", StaticFiles(directory="storage/output_feedback_frame"), name="static")

# 루트 엔드포인트 (선택 사항)
@app.get("/")
def read_root():
    logger = logging.getLogger(__name__)
    logger.info("Root endpoint accessed")
    return {"message": "Hello, World!"}

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise e

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
