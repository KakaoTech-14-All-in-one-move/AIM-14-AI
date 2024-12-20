# tests/vlm_model/test_utils/test_encoding_image.py

import pytest
import numpy as np
import subprocess
from fastapi import HTTPException
from vlm_model.utils.encoding_image import encode_image
from vlm_model.exceptions import ImageEncodingError
from unittest import mock

def test_encode_image_success():
    # 샘플 이미지 데이터 (256x256 RGB)
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    expected_size = (256, 256)
    expected_quality = 70

    # encode_image 호출
    result = encode_image(image, max_size=expected_size, quality=expected_quality)

    # 결과는 Base64 문자열이어야 함
    assert isinstance(result, str)
    # Base64 문자열은 특정 길이를 가짐 (여기서는 예시로 1자 이상을 확인)
    assert len(result) > 0

def test_encode_image_resize_failure(mocker):
    # 이미지 데이터가 아닌 입력
    image = "not_an_image"

    # cv2.resize을 모킹하여 예외 발생
    mocker.patch("cv2.resize", side_effect=Exception("Resize error"))

    with pytest.raises(ImageEncodingError) as excinfo:
        encode_image(image)

    assert "이미지 인코딩 중 서버 오류가 발생했습니다." in str(excinfo.value)

def test_encode_image_encoding_failure(mocker):
    # 샘플 이미지 데이터
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # cv2.imencode을 모킹하여 실패 반환
    mocker.patch("cv2.imencode", return_value=(False, None))

    with pytest.raises(ImageEncodingError) as excinfo:
        encode_image(image)

    assert "이미지 인코딩에 실패했습니다." in str(excinfo.value)

def test_encode_image_unexpected_exception(mocker):
    # 샘플 이미지 데이터
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # cv2.resize을 모킹하여 예외 발생
    mocker.patch("cv2.resize", side_effect=TypeError("Unexpected error"))

    with pytest.raises(ImageEncodingError) as excinfo:
        encode_image(image)

    assert "이미지 인코딩 중 서버 오류가 발생했습니다." in str(excinfo.value)
