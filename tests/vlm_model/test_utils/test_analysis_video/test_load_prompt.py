# tests/vlm_model/test_utils/test_analysis/test_load_user_prompt.py

import pytest
from unittest import mock
from pathlib import Path
from vlm_model.utils.analysis_video.load_prompt import load_user_prompt
from vlm_model.exceptions import PromptImportingError
import os

def test_load_user_prompt_success(mocker):
    """
    정상적인 상황에서 프롬프트 파일을 성공적으로 로드하는지 확인합니다.
    """
    # 환경 변수 모킹
    mocker.patch.dict(os.environ, {"PROMPT_PATH": "/fake/prompts/user_prompt.txt"})
    
    # Mock Path 객체
    mock_prompt_path = Path("/fake/prompts/user_prompt.txt")
    
    # Mock 파일 읽기
    mock_file = mocker.mock_open(read_data="This is a test prompt.")
    mocker.patch("pathlib.Path.open", mock_file)
    
    # 함수 호출
    result = load_user_prompt()
    
    # 검증
    mock_file.assert_called_once_with('r', encoding='utf-8')
    assert result == "This is a test prompt."

def test_load_user_prompt_file_not_found(mocker):
    """
    프롬프트 파일이 존재하지 않는 경우 PromptImportingError가 발생하는지 확인합니다.
    """
    # 환경 변수 모킹
    mocker.patch.dict(os.environ, {"PROMPT_PATH": "/fake/prompts/nonexistent_prompt.txt"})
    
    # Mock Path.open에서 FileNotFoundError 발생
    mocker.patch("pathlib.Path.open", mocker.mock_open(), side_effect=FileNotFoundError("No such file"))
    
    with pytest.raises(PromptImportingError) as excinfo:
        load_user_prompt()
    
    # 검증
    assert "프롬프트 파일을 찾을 수 없습니다" in str(excinfo.value)

def test_load_user_prompt_other_exception(mocker):
    """
    파일 열기 중 다른 예외가 발생하는 경우 PromptImportingError가 발생하는지 확인합니다.
    """
    # 환경 변수 모킹
    mocker.patch.dict(os.environ, {"PROMPT_PATH": "/fake/prompts/user_prompt.txt"})
    
    # Mock Path.open에서 IOError 발생
    mocker.patch("pathlib.Path.open", mocker.mock_open(), side_effect=IOError("Read error"))
    
    with pytest.raises(PromptImportingError) as excinfo:
        load_user_prompt()
    
    # 검증
    assert "프롬프트 파일을 로드하는 중 오류가 발생했습니다" in str(excinfo.value)
