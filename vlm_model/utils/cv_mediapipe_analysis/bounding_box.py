# vlm_model/utils/cv_mediapipe_analysis/bounding_box.py - 의논후 삭제 or 유지 결론

import cv2
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import mp_pose, mp_hands

def draw_bounding_boxes(frame, feedback, pose_landmarks, face_landmarks, hand_landmarks):
    """
    문제 영역에 Bounding Box를 그리고 잘못된 부분을 시각적으로 표시합니다.

    Args:
        frame: OpenCV 프레임.
        feedback: 분석된 피드백 결과 딕셔너리.
        pose_landmarks: Mediapipe Pose 랜드마크.
        face_landmarks: Mediapipe Face 랜드마크.
        hand_landmarks: Mediapipe Hand 랜드마크 리스트.

    Returns:
        frame: Bounding Box가 추가된 프레임.
    """
    frame_height, frame_width, _ = frame.shape

    # 자세 문제 표시
    if feedback["posture_score"] > 0.8:
        if pose_landmarks:
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_x, nose_y = int(nose.x * frame_width), int(nose.y * frame_height)
            cv2.rectangle(frame, (nose_x - 50, nose_y - 50), (nose_x + 50, nose_y + 50), (0, 0, 255), 2)
            cv2.putText(frame, "Posture Issue", (nose_x - 70, nose_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 시선 문제 표시
    if feedback["gaze_score"] > 0.5:
        if face_landmarks:
            left_eye = face_landmarks.landmark[33]  # 왼쪽 눈
            right_eye = face_landmarks.landmark[263]  # 오른쪽 눈
            left_eye_x, left_eye_y = int(left_eye.x * frame_width), int(left_eye.y * frame_height)
            right_eye_x, right_eye_y = int(right_eye.x * frame_width), int(right_eye.y * frame_height)
            cv2.rectangle(frame, (left_eye_x - 10, left_eye_y - 10), (right_eye_x + 10, right_eye_y + 10), (255, 0, 0), 2)
            cv2.putText(frame, "Gaze Issue", (left_eye_x - 50, left_eye_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 손 제스처 문제 표시
    if feedback["gestures_score"] > 0.5 or feedback["hand_raised"]:
        if hand_landmarks:
            for hand in hand_landmarks:
                wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x, wrist_y = int(wrist.x * frame_width), int(wrist.y * frame_height)
                cv2.rectangle(frame, (wrist_x - 50, wrist_y - 50), (wrist_x + 50, wrist_y + 50), (0, 255, 0), 2)
                cv2.putText(frame, "Gesture Issue", (wrist_x - 70, wrist_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 갑작스러운 움직임 표시
    if feedback["sudden_movement_score"] > 0.5:
        cv2.putText(frame, "Sudden Movement Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    return frame