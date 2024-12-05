# exceptions.py

class VideoImportingError(Exception):
    def __init__(self, message: str):
        self.message = message

class VideoProcessingError(Exception):
    def __init__(self, message: str):
        self.message = message

class ImageEncodingError(Exception):
    def __init__(self, message: str):
        self.message = message

class PromptImportingError(Exception):
    def __init__(self, message: str):
        self.message = message