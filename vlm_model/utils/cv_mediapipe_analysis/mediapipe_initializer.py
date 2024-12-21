# vlm_model/utils/cv_mediapipe_analysis/mediapipe_initializer.py

import mediapipe as mp

# Mediapipe 솔루션 초기화
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Mediapipe 객체 생성 (전역 한 번만 실행)

# Pose: 사람의 자세(관절 위치) 분석을 위한 Mediapipe 솔루션
pose = mp_pose.Pose(
    static_image_mode=False,           # False: 비디오 스트림에서 여러 프레임을 처리할 때 사용 (트래킹 가능)
    min_detection_confidence=0.5,      # 포즈 감지의 최소 신뢰도 (0.5 이상일 때만 랜드마크 감지)
    min_tracking_confidence=0.5        # 랜드마크 추적의 최소 신뢰도 (트래킹 실패 시 재감지 수행)
)

# FaceMesh: 얼굴의 세부적인 랜드마크(468개 점)를 검출하는 Mediapipe 솔루션
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,           # False: 비디오 스트림에서 실시간으로 얼굴 랜드마크를 감지
    max_num_faces=1,                   # 최대 감지할 얼굴의 수 (여기서는 1명으로 제한)
    min_detection_confidence=0.5,      # 얼굴 감지의 최소 신뢰도
    min_tracking_confidence=0.5        # 랜드마크 추적의 최소 신뢰도
)

# Hands: 손의 랜드마크(21개 점)를 검출하는 Mediapipe 솔루션
hands = mp_hands.Hands(
    static_image_mode=False,           # False: 비디오 스트림에서 실시간으로 손 랜드마크 감지
    max_num_hands=2,                   # 최대 감지할 손의 수 (여기서는 양손까지 지원)
    min_detection_confidence=0.3,      # 손 감지의 최소 신뢰도 (낮출수록 더 많이 감지하지만 정확도 감소)
    min_tracking_confidence=0.3        # 랜드마크 추적의 최소 신뢰도
)


# Mediapipe 초기화
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)