# AIM-14-AI-VLM (Pitching 영상 처리 엔진)

"Pitching" 플랫폼의 인공지능 모델을 활용하여 비디오 분석을 수행하여 발표에 대하여 피드백을 제공하는 모델 입니다.
 **FastAPI**와 **Swagger UI**를 사용하여 CV(Computer Vision) & VLM(Video Language Model)를 활용한 영상 처리 서버입니다.
 이를 **AWS EC2**에 **Docker**로 배포한 후 **프론트엔드(FE)** 애플리케이션과 연동하는 과정을 다룹니다. 서버는 영상 업로드를 처리하고, 피드백 데이터를 생성하여 프론트엔드에 제공합니다. 또한, **OpenCV**와 **FFmpeg**를 활용하여 영상 코덱 변환 및 비디오 길이 계산 등의 기능을 포함합니다.
 
---

## 목차

- [사전 요구사항](#사전-요구사항)
  - [Swagger UI 테스트용 요구사항](#swagger-ui-테스트용-요구사항)
  - [배포 (CI/CD)용 요구사항](#배포-cicd용-요구사항)
- [디렉토리 구조](#디렉토리-구조)
- [설치 및 배포](#설치-및-배포)
  - [1. 리포지토리 클론](#1-리포지토리-클론)
  - [2. 환경 변수 설정](#2-환경-변수-설정)
  - [3. FFmpeg 설치](#3-ffmpeg-설치)
  - [4. Docker 이미지 빌드](#4-docker-이미지-빌드)
  - [5. Docker 컨테이너 실행](#5-docker-컨테이너-실행)
  - [6. 로컬 테스트](#6-로컬-테스트)
    - [6.1 사전 요구사항](#61-사전-요구사항)
    - [6.2 리포지토리 클론](#62-리포지토리-클론)
    - [6.3 가상 환경 생성 (선택 사항)](#63-가상-환경-생성-선택-사항)
    - [6.4 의존성 설치](#64-의존성-설치)
    - [6.5 환경 변수 설정](#65-환경-변수-설정)
    - [6.6 서버 실행](#66-서버-실행)
  - [7 API 테스트](#7-api-테스트)
- [CORS 설정](#cors-설정)
  - [발생한 이슈](#발생한-이슈)
  - [해결 방법](#해결-방법)
- [영상 처리](#영상-처리)
  - [1. 비디오 길이 계산](#1-비디오-길이-계산)
  - [2. 코덱 변환 (H.264 → VP9)](#2-코덱-변환-h264--vp9)
- [영상 처리 오류 처리](#영상-처리-오류-처리)
- [사용법](#사용법)
  - [1. 영상 업로드](#1-영상-업로드)
  - [2. 피드백 조회](#2-피드백-조회)
- [추가 자료](#추가-자료)

---

### 사전 요구사항

#### Swagger UI 테스트용 요구사항

- **FastAPI**: 백엔드 프레임워크.
- **Swagger UI**: API 문서화 및 테스트 도구.
- **FFmpeg**: 비디오 코덱 변환 및 정보 추출 도구.
- **OpenCV**: 비디오 처리 및 분석 라이브러리.
- **requirements.txt**: 라이브러리 설치 파일.

---

#### 배포 (CI/CD)용 요구사항

- **프론트엔드 애플리케이션**: 백엔드 서버와 연동하는 클라이언트 애플리케이션.
- **AWS EC2 인스턴스**: FastAPI의 경우 8000번 포트 등 필요한 포트가 허용되도록 보안 그룹 설정.
- **Docker**: EC2 인스턴스에 설치 필요.

---

## 디렉토리 구조

```
├── LICENSE
├── README.md
├── Research
├── __pycache__
├── logging_config.json
├── logs
│   ├── access.log
│   └── app.log
├── main.py
├── prompt.txt
├── requirements.txt
├── storage
│   ├── input_video
│   └── output_feedback_frame
├── tests
│   ├── conftest.py
│   ├── test_main.py
│   └── vlm_model
│       ├── test_routers
│       │   ├── test_delete_files.py
│       │   ├── test_send_feedback.py
│       │   └── test_upload_video.py
│       └── test_utils
│           ├── test_analysis.py
│           ├── test_analysis_video
│           │   ├── test_load_prompt.py
│           │   └── test_parse_feedback.py
│           ├── test_cv_mediapipe_analysis
│           │   ├── test_analyze_mediapipe_main.py
│           │   ├── test_calculate_gesture.py
│           │   ├── test_calculate_hand_move.py
│           │   ├── test_gaze_analysis.py
│           │   ├── test_gesture_analysis.py
│           │   ├── test_mediapipe_initializer.py
│           │   ├── test_movement_analysis.py
│           │   └── test_posture_analysis.py
│           ├── test_download_video.py
│           ├── test_encoding_feedback_image.py
│           ├── test_encoding_image.py
│           ├── test_processing_video.py
│           ├── test_read_video.py
│           ├── test_video_codec_conversion.py
│           └── test_video_duration.py
└── vlm_model
    ├── README.md
    ├── __init__.py
    ├── __pycache__
    ├── config.py
    ├── constants
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── behaviors.py
    ├── routers
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── send_feedback.py
    │   └── upload_video.py
    ├── schemas
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── feedback.py
    ├── utils
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── analysis.py
    │   ├── download_video.py
    │   ├── encoding_image.py
    │   ├── read_video.py
    │   ├── user_prompt.py
    │   ├── video_duration.py
    │   └── visualization.py
    └── video_processor(test_local).py
```

---

## 설치 및 배포

### 1. 리포지토리 클론

```bash
git clone https://github.com/KakaoTech-14-All-in-one-move/AIM-14-AI-VLM.git
```

### 2. 환경 변수 설정

`.env` 파일을 생성하여 필요한 환경 변수를 설정합니다. 예:

```bash
OPENAI_API_KEY=your_openai_api_key
PROMPT_PATH=./prompt.txt
UPLOAD_DIR=storage/input_video
FEEDBACK_DIR=storage/output_feedback_frame
SENTRY_DSN=your_sentry_api_key
TRACE_SAMPLE_RATE=1.0

# 폰트 관련 환경 변수
FONT_DIR=fonts
FONT_FILE=NotoSans-VariableFont_wdth,wght.ttf
FONT_SIZE=15
```

---

### 3. FFmpeg 설치

FFmpeg는 비디오 코덱 변환에 필수적입니다. 사용 중인 운영체제에 맞게 설치하세요.

#### MacOS (Homebrew 사용)

```bash
brew install ffmpeg
```

#### Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows

1. [FFmpeg 공식 웹사이트](https://ffmpeg.org/download.html)에서 Windows용 바이너리를 다운로드합니다.
2. FFmpeg 설치 디렉토리를 시스템 경로에 추가합니다.

---

### 4. Docker 이미지 빌드

`Dockerfile`에 **FFmpeg**와 **OpenCV** 등의 필요한 모든 의존성이 포함되어 있는지 확인합니다.

```bash

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \


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

# Python 종속성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# OpenCV 런타임 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 빌드된 Python 패키지 복사
COPY --from=builder /usr/local /usr/local

# /app/logs 디렉토리 생성
RUN mkdir -p /app/logs

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-config", "logging_config.json"]
```

# Python 종속성 설치

```bash
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
```


### Docker 이미지 빌드

```bash
docker build -t vlm-video-processing .
```

---

### 5. Docker 컨테이너 실행

```bash
docker run -d -p 8000:8000 --name vlm-container vlm-video-processing
```

---

### 6. 로컬 테스트

#### 6.1 사전 요구사항

- Python 3.9 이상
- FFmpeg 설치 (코덱 변환 기능 사용 시 필요)
- OpenCV 설치

#### 6.2 리포지토리 클론

```bash
git clone https://github.com/KakaoTech-14-All-in-one-move/AIM-14-AI-VLM.git
```

#### 6.3 가상 환경 생성 (선택 사항)

```bash
python3 -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate  # Windows
```

#### 6.4 의존성 설치

```bash
pip install -r requirements.txt
```

#### 6.5 환경 변수 설정

```bash
UPLOAD_DIR=storage/input_video
FEEDBACK_DIR=storage/output_feedback_frame
```

#### 6.6 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-config logging_config.json
```

## 7 API 테스트

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)에서 Swagger UI로 API를 테스트할 수 있습니다.

---

## CORS 설정

### 발생한 이슈

CORS 정책으로 인해 프론트엔드와의 연결

 문제가 발생할 수 있습니다.

### 해결 방법

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 URL로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 영상 처리

### 1. 비디오 길이 계산

```python
import cv2

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration
```

---

### 2. 코덱 변환 (H.264 → VP9)

```python
import subprocess

def convert_to_vp9(input_path: str, output_path: str) -> bool:
    command = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libvpx-vp9", "-b:v", "1M",
        "-c:a", "libopus", output_path
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
```

---

## 사용법

### 1. 영상 업로드

```
POST api/video/receive-video/
```

### 2. 피드백 조회

```
GET /api/video/video-send-feedback/{video_id}/
```

---

## 추가 자료

- **FastAPI 문서**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Docker 문서**: [https://docs.docker.com/](https://docs.docker.com/)
- **FFmpeg 문서**: [https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)
- **OpenCV 문서**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Mediapipe 문서**: [https://google.github.io/mediapipe/](https://google.github.io/mediapipe/)
- **OpenAI Vision API 문서**: [https://platform.openai.com/docs/](https://platform.openai.com/docs/)
- **FastAPI CORS 미들웨어**: [https://fastapi.tiangolo.com/tutorial/cors/](https://fastapi.tiangolo.com/tutorial/cors/)

---

## .dockerignore

```dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.env
.git
.gitignore
*.md
*.ipynb
storage/
logs/
```

- **Python 캐시 파일:** `__pycache__`, `.pyc`, `.pyo`, `.pyd`
- **환경 파일:** `.env`
- **Git 관련 파일:** `.git`, `.gitignore`
- **문서 파일:** `*.md`, `*.ipynb`
- **데이터 및 로그 디렉토리:** `storage/`, `logs/`
- **기타:** `fonts/`, `htmlcov/`

---

## requirements.txt

```plaintext
openai
pillow
tqdm
opencv-python
python-dotenv
numpy
fastapi
uvicorn
python-multipart
python-json-logger
colorlog
sentry-sdk[fastapi]
mediapipe
pytest
pytest-cov
pytest-mock
httpx
```

---

## Dockerfile

- **베이스 이미지 선택**
    - `python:3.12-slim`: 가벼운 Python 3.12 이미지를 사용하여 최종 이미지 크기를 최소화합니다.
- **환경 변수 설정**
    - `PYTHONDONTWRITEBYTECODE=1`: Python이 `.pyc` 파일을 생성하지 않도록 설정.
    - `PYTHONUNBUFFERED=1`: Python 출력이 버퍼링되지 않고 즉시 터미널에 출력되도록 설정.
- **작업 디렉터리 설정**
    - `/app` 디렉터리를 작업 디렉터리로 설정합니다.
- **시스템 종속성 설치**
    - `build-essential`, `libffi-dev`, `libssl-dev`, `ffmpeg`, `libsndfile1` 등을 설치합니다.
    - `ffmpeg`: 오디오 및 비디오 처리에 필요합니다.
    - `libsndfile1`: 오디오 파일 처리를 위한 라이브러리입니다.
- **Python 종속성 설치**
    - `requirements.txt` 파일을 복사한 후, `pip`을 업그레이드하고 필요한 Python 패키지를 설치합니다.
- **프로젝트 파일 복사**
    - 현재 디렉터리의 모든 파일을 컨테이너의 `/app` 디렉터리로 복사합니다.

---
