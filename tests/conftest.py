# tests/conftest.py

import pytest
import sys
import os
import numpy as np
from pathlib import Path  # Path 임포트 추가
from fastapi.testclient import TestClient

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent  # 실제 디렉토리 구조에 맞게 조정
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from main import app  # FastAPI 앱이 정의된 파일을 import

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image():
    # 256x256 RGB 이미지 생성
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

@pytest.fixture
def sample_video_file(tmp_path):
    # 임시 비디오 파일 생성
    video_path = tmp_path / "sample_video.mp4"
    video_path.touch()
    return video_path
