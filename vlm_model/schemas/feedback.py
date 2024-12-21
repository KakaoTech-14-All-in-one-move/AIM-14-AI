# vlm_model/schemas/feedback.py

from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    video_id: str
    message: str

class FeedbackDetails(BaseModel):
    improvement: str # 개선이 필요한 점
    recommendations: str # 권장 사항

class FeedbackSections(BaseModel):
    gaze_processing: FeedbackDetails # 시선 처리
    facial_expression: FeedbackDetails # 얼굴 표정
    gestures: FeedbackDetails # 제스처 및 손동작
    posture_body: FeedbackDetails # 자세 및 신체 언어
    movement: FeedbackDetails # 갑작스러운 행동 및 움직임

class FeedbackFrame(BaseModel):
    video_id: str # video_id 추가
    frame_index: int
    timestamp: str  # 예: "0m 0s"
    feedback_text: FeedbackSections  # 피드백 섹션 구조로 변환된 텍스트
    image_base64: str  # Base64로 인코딩된 이미지 데이터

class FeedbackResponse(BaseModel):
    feedbacks: List[FeedbackFrame]
    message: str
    problem: Optional[str] = None  # 문제가 없을 때 추가

class DeleteResponse(BaseModel):
    video_id: str
    message: str