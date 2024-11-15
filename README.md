# AIM-14-AI (VLM Model)

"Pitching" 플랫폼의 인공지능 모델을 활용하여 비디오 분석을 수행하여 발표에 대하여 피드백을 제공하는 모델 입니다.
 **FastAPI**와 **Swagger UI**를 사용하여 VLM(Video Language Model) 영상 처리 서버를 개발하고, 이를 **AWS EC2**에 **Docker**로 배포한 후 **프론트엔드(FE)** 애플리케이션과 연동하는 과정을 다룹니다. 서버는 영상 업로드를 처리하고, 피드백 데이터를 생성하여 프론트엔드에 제공합니다. 또한, **OpenCV**와 **FFmpeg**를 활용하여 영상 코덱 변환 및 비디오 길이 계산 등의 기능을 포함합니다.

---

## 목차

- [사전 요구사항](#사전-요구사항)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 배포](#설치-및-배포)
  - [1. 리포지토리 클론](#1-리포지토리-클론)
  - [2. 환경 변수 설정](#2-환경-변수-설정)
  - [3. Docker 이미지 빌드](#3-docker-이미지-빌드)
  - [4. Docker 컨테이너 실행](#4-docker-컨테이너-실행)
- [CORS 설정](#cors-설정)
  - [발생한 이슈](#발생한-이슈)
  - [해결 방법](#해결-방법)
- [영상 처리](#영상-처리)
  - [1. 비디오 길이 계산](#1-비디오-길이-계산)
  - [2. 코덱 변환 (H.264 -> VP9)](#2-코덱-변환-h264---vp9)
  - [3. Google Colab을 활용한 테스트](#3-google-colab을-활용한-테스트)
- [영상 처리 오류 처리](#영상-처리-오류-처리)
- [트러블슈팅](#트러블슈팅)
  - [1. 영상 길이를 가져올 수 없음](#1-영상-길이를-가져올-수-없음)
  - [2. 프론트엔드에서의 잘못된 HTTP 요청](#2-프론트엔드에서의-잘못된-http-요청)
- [사용법](#사용법)
  - [영상 업로드](#영상-업로드)
  - [피드백 조회](#피드백-조회)
- [로깅](#로깅)
- [베스트 프랙티스](#베스트-프랙티스)
- [추가 자료](#추가-자료)
- [문의](#문의)

---

## 사전 요구사항

- **AWS EC2 인스턴스**: FastAPI의 경우 8000번 포트 등 필요한 포트가 허용되도록 보안 그룹 설정.
- **Docker**: EC2 인스턴스에 설치 필요.
- **FastAPI**: 백엔드 프레임워크.
- **Swagger UI**: API 문서화 및 테스트 도구.
- **프론트엔드 애플리케이션**: 백엔드 서버와 연동하는 클라이언트 애플리케이션.
- **FFmpeg**: 비디오 코덱 변환 및 정보 추출 도구.
- **OpenCV**: 비디오 처리 및 분석 라이브러리.

---

## 프로젝트 구조

```
vlm_model/
├── routers/
│   ├── upload_video.py
│   └── send_feedback.py
├── schemas/
│   └── feedback.py
├── utils/
│   ├── video_duration.py
│   ├── download_video.py
│   ├── analysis.py
│   ├── encoding_image.py
│   └── convert_codec.py
├── config.py
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 설치 및 배포

### 1. 리포지토리 클론

```bash
git clone https://github.com/your-username/vlm_video_processing.git
cd vlm_video_processing
```

### 2. 환경 변수 설정

`.env` 파일을 생성하여 필요한 환경 변수를 설정합니다. 예:

```bash
OPENAI_API_KEY=API_KEY
PROMPT_PATH=./prompt.txt
```

해당 디렉터리가 존재하지 않으면 배포 중 자동으로 생성됩니다.

### 3. Docker 이미지 빌드

`Dockerfile`에 **FFmpeg**와 **OpenCV** 등의 필요한 모든 의존성이 포함되어 있는지 확인합니다.

**Dockerfile 예시:**

```dockerfile
FROM python:3.9-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libvpx6 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker 이미지 빌드:**

```bash
docker build -t vlm-video-processing .
```

### 4. Docker 컨테이너 실행

```bash
docker run -d -p 8000:8000 --name vlm-container vlm-video-processing
```

포트가 올바르게 매핑되고 EC2 보안 그룹에서 포트 `8000`으로의 인바운드 트래픽이 허용되어 있는지 확인하세요.

---

## CORS 설정

### 발생한 이슈

FastAPI 서버를 AWS EC2에 Docker로 배포한 후 프론트엔드와 연결할 때 **CORS (Cross-Origin Resource Sharing)** 이슈가 발생할 수 있습니다.

**오류 예시:**

```
CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

### 해결 방법

FastAPI에서 CORS 이슈를 해결하기 위해 **`CORSMiddleware`**를 사용하여 미들웨어를 설정할 수 있습니다.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 도메인 또는 IP로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- **운영 환경**에서는 보안을 위해 특정 출처만 허용하도록 설정하는 것이 권장됩니다.

---

## 영상 처리

### 1. 비디오 길이 계산

비디오 파일의 길이를 초 단위로 계산하기 위해 OpenCV와 FFmpeg를 활용합니다.

**`utils/video_duration.py` 예시:**

```python
import cv2
from typing import Optional

def get_video_duration(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()
    return duration
```

### 2. 코덱 변환 (H.264 → VP9)

H.264 코덱으로 인코딩된 비디오를 VP9 코덱으로 변환하기 위해 FFmpeg를 사용합니다.

**`utils/convert_codec.py` 예시:**

```python
import subprocess

def convert_to_vp9(input_path: str, output_path: str) -> bool:
    command = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libvpx-vp9', '-b:v', '1M',
        '-c:a', 'libopus', output_path
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
```
---

## 영상 처리 오류 처리

다양한 오류 상황을 대비하여 아래와 같은 예외 처리를 추가했습니다:

- **파일 없음**: `HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")`
- **지원되지 않는 형식**: 허용된 파일 형식을 안내하는 응답 반환.
- **FFmpeg 또는 코덱 문제**: `HTTPException(status_code=500, detail="코덱 변환 실패")`

---

## 사용법

### 영상 업로드

**엔드포인트:**

```
POST /api/video/receive-video/
```

### 피드백 조회

**엔드포인트:**

```
GET /api/video/video-send-feedback/{video_id}/
```

---

## 로깅

서버는 `logging` 모듈을 사용하여 중요한 이벤트와 오류를 기록합니다.

---

## 추가 자료

- **FastAPI 문서**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Docker 문서**: [https://docs.docker.com/](https://docs.docker.com/)
- **FFmpeg 문서**: [https://ffmpeg.org/documentation.html/](https://ffmpeg.org/documentation.html/)
- **OpenCV 문서**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **FastAPI CORS 미들웨어**: [FastAPI CORS Middleware](https://fastapi.tiangolo.com/tutorial/cors/)

---
