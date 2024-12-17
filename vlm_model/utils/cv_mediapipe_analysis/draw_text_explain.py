# vlm_model/utils/cv_mediapipe_analysis/draw_text_explain.py

import cv2

def draw_text(frame, text, x, y, color=(0, 255, 0)):
    """
    프레임 위에 텍스트를 그립니다.

    Args:
        frame: OpenCV 프레임
        text: 표시할 텍스트
        x, y: 텍스트 좌표
        color: 텍스트 색상 (BGR)
    """
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)