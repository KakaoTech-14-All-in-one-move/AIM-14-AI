# Python 베이스 이미지 선택
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# libGL 설치 (Ubuntu 기반의 Python 이미지에 맞게 설치)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 환경 변수 설정 (Docker 전용 설정)
ENV DOCKER_ENV=1

# 로컬 코드와 requirements.txt 복사
COPY . .

# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# uvicorn 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
