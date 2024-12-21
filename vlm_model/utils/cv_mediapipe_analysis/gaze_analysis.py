# vlm_model/utils/cv_mediapipe_analysis/gaze_analysis.py

import numpy as np

def calculate_lack_of_eye_contact_score(face_landmarks, frame_width):
    """
    발표자의 시선 부족 정도를 점수로 반환합니다.

    Args:
        face_landmarks: Mediapipe FaceMesh 랜드마크.
        frame_width: 프레임 너비.

    Returns:
        float: 시선 부족 점수 (0에서 1 사이).
    """
    LEFT_EYE = 33
    RIGHT_EYE = 263

    # 왼쪽/오른쪽 눈의 x좌표 구하기
    left_eye_x = face_landmarks.landmark[LEFT_EYE].x * frame_width
    right_eye_x = face_landmarks.landmark[RIGHT_EYE].x * frame_width

    # 화면 중앙 범위 설정
    center_min = 0.4 * frame_width
    center_max = 0.6 * frame_width

    # 눈 위치가 화면 중앙에서 얼마나 벗어났는지 계산
    left_deviation = max(center_min - left_eye_x, left_eye_x - center_max, 0)
    right_deviation = max(center_min - right_eye_x, right_eye_x - center_max, 0)

    # 편차를 정규화하여 점수 환산
    gaze_score = (left_deviation + right_deviation) / frame_width
    return round(min(gaze_score / 0.1, 1.0), 2)  # 소수점 두 자리로 제한