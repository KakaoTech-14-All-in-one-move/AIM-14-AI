# pronun_model/exceptions.py

class VideoImportingError(Exception):
    """
    비디오 파일을 임포트하는 중 발생하는 예외.

    Attributes:
        message (str): 예외에 대한 상세 메시지.
    """
    def __init__(self, message: str):
        self.message = message


class VideoProcessingError(Exception):
    """
    비디오 파일을 처리하는 중 발생하는 예외.

    Attributes:
        message (str): 예외에 대한 상세 메시지.
    """
    def __init__(self, message: str):
        self.message = message


class ImageEncodingError(Exception):
    """
    이미지를 인코딩하는 중 발생하는 예외.

    Attributes:
        message (str): 예외에 대한 상세 메시지.
    """
    def __init__(self, message: str):
        self.message = message


class PromptImportingError(Exception):
    """
    프롬프트 파일을 임포트하는 중 발생하는 예외.

    Attributes:
        message (str): 예외에 대한 상세 메시지.
    """
    def __init__(self, message: str):
        self.message = message
