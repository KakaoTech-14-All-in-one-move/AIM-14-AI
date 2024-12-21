from setuptools import setup, find_packages

setup(
    name='AIM-14-AI-VLM',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "openai",
        "pillow",
        "tqdm",
        "opencv-python",
        "python-dotenv",
        "numpy",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "python-json-logger",
        "colorlog",
        "sentry-sdk[fastapi]",
        "mediapipe",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "httpx",
    ],
)
