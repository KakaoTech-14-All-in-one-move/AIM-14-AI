# tests/vlm_model/test_utils/test_video_codec_conversion.py

from fastapi import HTTPException
import subprocess
import numpy as np
import pytest
from unittest import mock
from vlm_model.utils.video_codec_conversion import convert_to_vp9_if_needed, convert_to_vp9, is_vp9, get_video_codec_info
from vlm_model.exceptions import VideoImportingError

def test_convert_to_vp9_if_needed_already_vp9(mocker):
    input_path = "/fake/input_vp9.webm"
    output_path = "/fake/output_vp9.webm"

    # is_vp9 함수를 모킹하여 True 반환
    mocker.patch("vlm_model.utils.video_codec_conversion.is_vp9", return_value=True)

    result = convert_to_vp9_if_needed(input_path, output_path)
    assert result == False

def test_convert_to_vp9_if_needed_not_vp9_success(mocker):
    input_path = "/fake/input.mp4"
    output_path = "/fake/output.webm"

    # is_vp9 함수를 모킹하여 False 반환
    mocker.patch("vlm_model.utils.video_codec_conversion.is_vp9", return_value=False)
    # convert_to_vp9 함수를 모킹하여 True 반환
    mock_convert = mocker.patch("vlm_model.utils.video_codec_conversion.convert_to_vp9", return_value=True)

    result = convert_to_vp9_if_needed(input_path, output_path)
    assert result == True
    mock_convert.assert_called_once_with(
        input_path=input_path,
        output_path=output_path,
        preset='faster',
        cpu_used=8,
        threads=0,
        tile_columns=4,
        tile_rows=2,
        bitrate='1M'
    )

def test_convert_to_vp9_if_needed_conversion_failure(mocker):
    input_path = "/fake/input.mp4"
    output_path = "/fake/output.webm"

    # is_vp9 함수를 모킹하여 False 반환
    mocker.patch("vlm_model.utils.video_codec_conversion.is_vp9", return_value=False)
    # convert_to_vp9 함수를 모킹하여 예외 발생
    mocker.patch("vlm_model.utils.video_codec_conversion.convert_to_vp9", side_effect=VideoImportingError("Conversion failed"))

    with pytest.raises(VideoImportingError) as excinfo:
        convert_to_vp9_if_needed(input_path, output_path)

    assert "비디오 변환 중 오류가 발생했습니다." in str(excinfo.value)

def test_convert_to_vp9_success(mocker):
    input_path = "/fake/input.mp4"
    output_path = "/fake/output.webm"

    # subprocess.run을 모킹하여 성공적으로 실행
    mock_subprocess = mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run", return_value=mock.Mock())

    result = convert_to_vp9(input_path, output_path)
    assert result == True
    mock_subprocess.assert_called_once()

def test_convert_to_vp9_file_not_found(mocker):
    input_path = "/fake/input.mp4"
    output_path = "/fake/output.webm"

    # subprocess.run을 모킹하여 FileNotFoundError 발생
    mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run", side_effect=FileNotFoundError("ffmpeg not found"))

    with pytest.raises(HTTPException) as excinfo:
        convert_to_vp9(input_path, output_path)

    assert "ffmpeg 패키지 파일을 찾을수 없습니다. 설치가 필요합니다." in str(excinfo.value)
    assert excinfo.value.status_code == 404

def test_convert_to_vp9_called_process_error(mocker):
    input_path = "/fake/input.mp4"
    output_path = "/fake/output.webm"

    # subprocess.run을 모킹하여 CalledProcessError 발생
    mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg"))

    with pytest.raises(VideoImportingError) as excinfo:
        convert_to_vp9(input_path, output_path)

    assert "비디오 변환 중 오류가 발생했습니다." in str(excinfo.value)

def test_is_vp9_true(mocker):
    video_path = "/fake/video.vp9.webm"
    mocker.patch("vlm_model.utils.video_codec_conversion.get_video_codec_info", return_value="VP9")
    assert is_vp9(video_path) == True

def test_is_vp9_false(mocker):
    video_path = "/fake/video.mp4"
    mocker.patch("vlm_model.utils.video_codec_conversion.get_video_codec_info", return_value="H264")
    assert is_vp9(video_path) == False

def test_get_video_codec_info_success(mocker):
    video_path = "/fake/video.mp4"
    ffmpeg_output = "Stream #0:0: Video: H264 (High), yuv420p(progressive), 1280x720, 30 fps, 30 tbr, 30 tbn, 60 tbc"

    # subprocess.run을 모킹하여 성공적으로 실행
    mock_run = mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run")
    mock_run.return_value.stderr = ffmpeg_output.encode()

    result = get_video_codec_info(video_path)
    assert result == "H264"

def test_get_video_codec_info_no_match(mocker):
    video_path = "/fake/video.unknown"
    ffmpeg_output = "Stream #0:0: Video: UnknownCodec, yuv420p, 1280x720, 30 fps"

    # subprocess.run을 모킹하여 성공적으로 실행
    mock_run = mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run")
    mock_run.return_value.stderr = ffmpeg_output.encode()

    result = get_video_codec_info(video_path)
    assert result == ""

def test_get_video_codec_info_failure(mocker):
    video_path = "/fake/video.mp4"

    # subprocess.run을 모킹하여 CalledProcessError 발생
    mock_run = mocker.patch("vlm_model.utils.video_codec_conversion.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg"))

    with pytest.raises(VideoImportingError) as excinfo:
        get_video_codec_info(video_path)

    assert "코덱 정보 확인 실패" in str(excinfo.value)
