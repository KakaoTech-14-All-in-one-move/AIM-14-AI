# setup.py

from setuptools import setup, find_packages

setup(
    name='AIM-14-AI-VLM',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "pytest",
        "pytest-mock",
        "mediapipe",
        "opencv-python",
        "numpy",
        "python-dotenv",
        # 기타 필요한 패키지들
    ],
)
