# 공식 Python 경량 이미지를 베이스로 사용
FROM python:3.12-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# /app/logs 디렉토리 생성
RUN mkdir -p /app/logs

# Python 종속성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# PYTHONPATH 설정
ENV PYTHONPATH=/app

# FastAPI가 실행될 포트 개방
EXPOSE 8000

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-config", "logging_config.json"]