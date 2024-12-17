# vlm_model/utils/cv_mediapipe_analysis/calculate.py

from vlm_model.utils.cv_mediapipe_analysis.analyze_mediapipe_main import analyze_frame

def analyze_video_frames(frames, posture_threshold=0.8, gaze_threshold=0.5,
                         gesture_threshold=0.5, movement_threshold=0.5,
                         hand_movement_threshold=0.5):
    """
    여러 프레임(영상)을 순차적으로 분석하여 결과를 반환합니다.
    각 임계값을 초과하거나 특정 조건(hand_out_of_frame, hand_raised)을 만족하면 프레임과 결과를 추출합니다.

    Args:
        frames: 분석할 프레임 리스트.
        posture_threshold: 자세 불량 판정 기준 점수 - posture_body.
        gaze_threshold: 시선 부족 판정 기준 점수 - gaze_processing.
        gesture_threshold: 손 제스처 과도 판정 기준 점수 - gestures.
        movement_threshold: 갑작스런 움직임 판정 기준 점수 - movement.
        hand_movement_threshold: 손 움직임 과도 판정 기준 점수 - gestures.

    Returns:
        filtered_feedback_results: 기준 초과 또는 조건 만족 피드백 결과 리스트.
        filtered_frames: 기준 초과 또는 조건 만족 프레임 리스트.
    """
    filtered_feedback_results = []
    filtered_frames = []
    previous_pose_landmarks = None
    previous_hand_landmarks = None

    for idx, frame in enumerate(frames):
        # 각 프레임을 분석
        feedback, current_pose_landmarks, current_hand_landmarks = analyze_frame(
            frame, previous_pose_landmarks, previous_hand_landmarks
        )

        # 기준 초과 여부 또는 특정 조건 만족 여부 확인
        if (feedback["posture_score"] > posture_threshold or
            feedback["gaze_score"] > gaze_threshold or
            feedback["gestures_score"] > gesture_threshold or
            feedback["sudden_movement_score"] > movement_threshold or
            feedback["hand_out_of_frame"] or  # 손이 화면 밖으로 나간 경우
            feedback["hand_raised"]):         # 손이 너무 높이 올라간 경우

            # 기준 초과 또는 조건 만족 프레임 저장
            feedback["frame_index"] = idx
            filtered_feedback_results.append(feedback)
            filtered_frames.append(frame)

        # 이전 랜드마크 업데이트
        previous_pose_landmarks = current_pose_landmarks if current_pose_landmarks else previous_pose_landmarks
        previous_hand_landmarks = current_hand_landmarks if current_hand_landmarks else previous_hand_landmarks

    return filtered_feedback_results, filtered_frames
