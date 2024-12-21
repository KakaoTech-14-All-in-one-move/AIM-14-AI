# tests/vlm_model/test_utils/test_analysis_video/test_parse_feedback.py

import pytest
from vlm_model.utils.analysis_video.parse_feedback import parse_feedback_text
from vlm_model.schemas.feedback import FeedbackSections, FeedbackDetails
from vlm_model.exceptions import VideoProcessingError
from fastapi import HTTPException

def test_parse_feedback_text_success_with_problem():
    feedback_text = """```json
{
    "gaze_processing": {
        "improvement": "Improve gaze towards the screen.",
        "recommendations": "Maintain eye contact with the audience."
    },
    "facial_expression": {
        "improvement": "Smiling more often.",
        "recommendations": "Avoid looking bored."
    },
    "gestures": {
        "improvement": "Use more hand gestures.",
        "recommendations": "Avoid repetitive movements."
    },
    "posture_body": {
        "improvement": "Stand straight.",
        "recommendations": "Avoid slouching."
    },
    "movement": {
        "improvement": "Move around the stage.",
        "recommendations": "Avoid pacing back and forth."
    }
}
```"""

    expected_output = FeedbackSections(
        gaze_processing=FeedbackDetails(improvement="Improve gaze towards the screen.", recommendations="Maintain eye contact with the audience."),
        facial_expression=FeedbackDetails(improvement="Smiling more often.", recommendations="Avoid looking bored."),
        gestures=FeedbackDetails(improvement="Use more hand gestures.", recommendations="Avoid repetitive movements."),
        posture_body=FeedbackDetails(improvement="Stand straight.", recommendations="Avoid slouching."),
        movement=FeedbackDetails(improvement="Move around the stage.", recommendations="Avoid pacing back and forth.")
    )

    result = parse_feedback_text(feedback_text)
    assert result == expected_output

def test_parse_feedback_text_success_no_problem():
    feedback_text = """```json
{
    "problem": "none"
}
```"""

    expected_output = FeedbackSections(
        gaze_processing=FeedbackDetails(improvement="", recommendations=""),
        facial_expression=FeedbackDetails(improvement="", recommendations=""),
        gestures=FeedbackDetails(improvement="", recommendations=""),
        posture_body=FeedbackDetails(improvement="", recommendations=""),
        movement=FeedbackDetails(improvement="", recommendations="")
    )

    result = parse_feedback_text(feedback_text)
    assert result == expected_output

def test_parse_feedback_text_no_code_block():
    feedback_text = """
{
    "gaze_processing": {
        "improvement": "Improve gaze towards the screen.",
        "recommendations": "Maintain eye contact with the audience."
    },
    "facial_expression": {
        "improvement": "Smiling more often.",
        "recommendations": "Avoid looking bored."
    },
    "gestures": {
        "improvement": "Use more hand gestures.",
        "recommendations": "Avoid repetitive movements."
    },
    "posture_body": {
        "improvement": "Stand straight.",
        "recommendations": "Avoid slouching."
    },
    "movement": {
        "improvement": "Move around the stage.",
        "recommendations": "Avoid pacing back and forth."
    }
}
"""

    expected_output = FeedbackSections(
        gaze_processing=FeedbackDetails(improvement="Improve gaze towards the screen.", recommendations="Maintain eye contact with the audience."),
        facial_expression=FeedbackDetails(improvement="Smiling more often.", recommendations="Avoid looking bored."),
        gestures=FeedbackDetails(improvement="Use more hand gestures.", recommendations="Avoid repetitive movements."),
        posture_body=FeedbackDetails(improvement="Stand straight.", recommendations="Avoid slouching."),
        movement=FeedbackDetails(improvement="Move around the stage.", recommendations="Avoid pacing back and forth.")
    )

    result = parse_feedback_text(feedback_text)
    assert result == expected_output

def test_parse_feedback_text_empty_feedback_text():
    feedback_text = ""

    with pytest.raises(VideoProcessingError) as excinfo:
        parse_feedback_text(feedback_text)
    
    assert "비어있는 피드백 텍스트가 전달되었습니다." in str(excinfo.value)

def test_parse_feedback_text_invalid_json():
    feedback_text = """```json
{
    "gaze_processing": {
        "improvement": "Improve gaze towards the screen.",
        "recommendations": "Maintain eye contact with the audience.",
    }  # Trailing comma makes it invalid JSON
}
```"""

    with pytest.raises(HTTPException) as excinfo:
        parse_feedback_text(feedback_text)
    
    assert excinfo.value.status_code == 400
    assert "JSON 디코딩 오류" in excinfo.value.detail

def test_parse_feedback_text_missing_sections():
    feedback_text = """```json
{
    "gaze_processing": {
        "improvement": "Improve gaze towards the screen.",
        "recommendations": "Maintain eye contact with the audience."
    }
}
```"""

    expected_output = FeedbackSections(
        gaze_processing=FeedbackDetails(improvement="Improve gaze towards the screen.", recommendations="Maintain eye contact with the audience."),
        facial_expression=FeedbackDetails(improvement="", recommendations=""),
        gestures=FeedbackDetails(improvement="", recommendations=""),
        posture_body=FeedbackDetails(improvement="", recommendations=""),
        movement=FeedbackDetails(improvement="", recommendations="")
    )

    result = parse_feedback_text(feedback_text)
    assert result == expected_output

def test_parse_feedback_text_unexpected_exception(mocker):
    feedback_text = """```json
{
    "gaze_processing": {
        "improvement": "Improve gaze towards the screen.",
        "recommendations": "Maintain eye contact with the audience."
    }
}
```"""

    # Mock json.loads to raise an unexpected exception
    mocker.patch("vlm_model.utils.analysis_video.parse_feedback.json.loads", side_effect=Exception("Unexpected error"))

    with pytest.raises(HTTPException) as excinfo:
        parse_feedback_text(feedback_text)
    
    assert excinfo.value.status_code == 500
    assert "FeedbackSections 생성 실패" in excinfo.value.detail
