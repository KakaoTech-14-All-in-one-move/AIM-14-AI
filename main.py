# main.py
import sys
import os

# 현재 파일의 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from vlm_model.routers.upload_video import router as upload_video_router
import uvicorn
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:5500",  # 예: 로컬 개발 서버
    # 필요에 따라 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 실제 사용 시 필요한 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(upload_video_router, prefix="/api/video", tags=["Video Upload"])

# 루트 엔드포인트 (선택 사항)
@app.get("/")
def read_root():
    return {"message": "VLM Model GPT-4O-Mini API"}

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

# 서버 실행 (optional, can also run using uvicorn from command line)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)