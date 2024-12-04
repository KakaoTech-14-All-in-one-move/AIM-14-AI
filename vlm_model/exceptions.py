# exceptions.py

class VideoProcessingError(Exception):
    def __init__(self, message: str):
        self.message = message

class ImageEncodingError(Exception):
    def __init__(self, message: str):
        self.message = message