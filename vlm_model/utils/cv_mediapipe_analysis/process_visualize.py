# vlm_model/utils/cv_mediapipe_analysis/process_visualize.py - 삭제 예정?

from vlm_model.utils.cv_mediapipe_analysis.bounding_box import draw_bounding_boxes

def process_feedback_and_visualize(frames, feedback_results):
    """
    분석된 피드백 결과를 기반으로 잘못된 부분을 시각적으로 표시합니다.

    Args:
        frames: 비디오 프레임 리스트.
        feedback_results: 각 프레임의 피드백 결과 리스트.

    Returns:
        processed_frames: Bounding Box와 텍스트가 추가된 프레임 리스트.
    """
    processed_frames = []
    for idx, (frame, feedback) in enumerate(zip(frames, feedback_results)):
        if feedback["frame_index"] == idx:
            pose_landmarks = feedback.get("pose_landmarks", None)
            face_landmarks = feedback.get("face_landmarks", None)
            hand_landmarks = feedback.get("hand_landmarks", None)
            frame = draw_bounding_boxes(frame, feedback, pose_landmarks, face_landmarks, hand_landmarks)
            processed_frames.append(frame)
    return processed_frames