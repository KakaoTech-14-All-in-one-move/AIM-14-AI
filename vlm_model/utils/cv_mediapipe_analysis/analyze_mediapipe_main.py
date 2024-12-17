# vlm_model/utils/cv_mediapipe_analysis/analyze_mediapipe_main.py

# Mediapipe 초기화
from vlm_model.utils.cv_mediapipe_analysis.mediapipe_initializer import pose, face_mesh, hands

from vlm_model.utils.cv_mediapipe_analysis.posture_analysis import calculate_head_position_score
from vlm_model.utils.cv_mediapipe_analysis.movement_analysis import calculate_sudden_movement_score
from vlm_model.utils.cv_mediapipe_analysis.gaze_analysis import calculate_lack_of_eye_contact_score
from vlm_model.utils.cv_mediapipe_analysis.gesture_analysis import calculate_excessive_gestures_score
from vlm_model.utils.cv_mediapipe_analysis.calculate_hand_move import calculate_hand_movement_score
from vlm_model.utils.cv_mediapipe_analysis.calculate_gesture import calculate_gestures_score
import mediapipe as mp

import logging
import cv2
from typing import Tuple, Dict, Optional

# 모듈별 로거 생성
logger = logging.getLogger(__name__)

def analyze_frame(frame: cv2.Mat, previous_pose_landmarks: Optional[object] = None, previous_hand_landmarks: Optional[object] = None) -> Tuple[Dict[str, float], Optional[object], Optional[object]]:
    """
    단일 프레임을 분석하여 자세, 시선, 손동작 등의 피드백 정보를 반환합니다.

    Args:
        frame: 분석할 OpenCV 프레임.
        previous_pose_landmarks: 이전 프레임의 포즈 랜드마크 - gestures.
        previous_hand_landmarks: 이전 프레임의 손 랜드마크 - gestures.

    Returns:
        feedback: 분석 결과를 담은 딕셔너리.
        current_pose_landmarks: 현재 포즈 랜드마크.
        current_hand_landmarks: 현재 손 랜드마크.
    """
    try:
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe 결과 처리
        pose_results = pose.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        logger.debug(f"Pose results: {pose_results.pose_landmarks if pose_results.pose_landmarks else 'None'}")
        logger.debug(f"Face results: {face_results.multi_face_landmarks if face_results.multi_face_landmarks else 'None'}")
        logger.debug(f"Hand results: {hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else 'None'}")

        # 초기 점수들
        posture_score = 0.0
        gaze_score = 0.0
        excessive_gestures_score = 0.0
        sudden_movement_score = 0.0
        hand_movement_score = 0.0

        current_pose_landmarks = None
        current_hand_landmarks = None

        # 포즈 분석
        if pose_results.pose_landmarks:
            current_pose_landmarks = pose_results.pose_landmarks
            posture_score = calculate_head_position_score(current_pose_landmarks, frame_width, frame_height)
            sudden_movement_score = calculate_sudden_movement_score(current_pose_landmarks, previous_pose_landmarks)

        # 얼굴(시선) 분석
        if face_results.multi_face_landmarks:
            gaze_score = calculate_lack_of_eye_contact_score(face_results.multi_face_landmarks[0], frame_width)

        # 손 분석
        if hand_results.multi_hand_landmarks:
            current_hand_landmarks = hand_results.multi_hand_landmarks[0]
            for hl in hand_results.multi_hand_landmarks:
                g_score = calculate_excessive_gestures_score(hl)
                if g_score > excessive_gestures_score:
                    excessive_gestures_score = g_score

                if previous_hand_landmarks is not None:
                    m_score = calculate_hand_movement_score(hl, previous_hand_landmarks)
                    if m_score > hand_movement_score:
                        hand_movement_score = m_score

        # gestures 평균 점수 계산
        gestures_score = calculate_gestures_score(excessive_gestures_score, hand_movement_score)

        # 피드백 딕셔너리
        feedback = {
            "posture_score": round(posture_score, 2),
            "gaze_score": round(gaze_score, 2),
            "gestures_score": round(gestures_score, 2),
            "sudden_movement_score": round(sudden_movement_score, 2)
        }

        return feedback, current_pose_landmarks, current_hand_landmarks

    except Exception as e:
        logger.error(f"Mediapipe로 CV 분석 중 오류 발생: {str(e)}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        # 기본값 반환
        return {
            "posture_score": 0.0,
            "gaze_score": 0.0,
            "gestures_score": 0.0,
            "sudden_movement_score": 0.0
        }, None, None
