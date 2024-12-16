# vlm_model/utils/cv_mediapipe_analysis/draw_debug.py

import cv2
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_pose

def draw_debug_info(frame, pose_landmarks, frame_width, frame_height, posture_score):
    """
    디버깅을 위한 시각화 함수. 코 위치와 화면 중심선을 표시하고 자세 점수를 보여줍니다.

    Args:
        frame: OpenCV 프레임.
        pose_landmarks: Mediapipe Pose 랜드마크.
        frame_width: 프레임 너비.
        frame_height: 프레임 높이.
        posture_score: 계산된 자세 점수.
    """
    NOSE = mp_pose.PoseLandmark.NOSE.value
    nose_x = int(pose_landmarks.landmark[NOSE].x * frame_width)
    nose_y = int(pose_landmarks.landmark[NOSE].y * frame_height)

    # 머리(코) 위치 표시
    cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

    # 화면 중심선 그리기
    center_x, center_y = frame_width // 2, frame_height // 2
    cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
    cv2.line(frame, (0, center_y), (frame_width, center_y), (255, 0, 0), 2)

    # 자세 점수 표시
    cv2.putText(frame, f"Posture Score: {posture_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 자세가 매우 나쁘다면 추가 메시지
    if posture_score > 0.7:
        cv2.putText(frame, "Poor Posture", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)