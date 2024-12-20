# tests/conftest.py

import pytest
from fastapi.testclient import TestClient
from main import app  # FastAPI 앱이 정의된 파일을 import

@pytest.fixture
def client():
    return TestClient(app)
