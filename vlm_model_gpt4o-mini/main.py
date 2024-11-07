# main.py

import math
from config import OPENAI_KEY, SYSTEM_INSTRUCTION
from constants import PROBLEMATIC_BEHAVIORS
from utils import (
    read_video_opencv,
    get_video_duration,
    download_and_sample_video_local,
    analyze_frames,
    plot_problematic_frames,
    encode_image,
    generate_user_prompt
)
import openai

def main():
    # OpenAI API 키 설정
    openai.api_key = OPENAI_KEY  # client = OpenAI(api_key=OPENAI_KEY) # 비디오에서 특정 프레임을 추출하는 함수

    video_path = "/Users/daehyunkim_kakao/Desktop/Kakao Business (Project)/AIM-14-AI/video_storage/test_video.webm"  # 비디오 파일의 경로
    segment_length = 60  # 각 세그먼트의 길이햣 (초 단위)
    frame_interval = 3   # 프레임 추출 간격 (초 단위)

    # 비디오의 전체 길이(초 단위)를 가져옵니다.
    video_duration = get_video_duration(video_path)

    if video_duration is None:
        print("비디오 길이를 가져올 수 없습니다.")
        return

    # 비디오를 세그먼트로 분할하기 위한 세그먼트 수 계산
    num_segments = math.ceil(video_duration / segment_length)
    print(f"비디오 전체 길이: {int(video_duration // 60)}분 {int(video_duration % 60)}초")
    print(f"총 {num_segments}개의 세그먼트로 분할됩니다.")

    all_segments_frames = []  # 모든 세그먼트의 프레임들을 저장할 리스트
    all_durations = []        # 각 세그먼트의 지속 시간을 저장할 리스트

    # 각 세그먼트별로 프레임을 추출합니다.
    for i in range(num_segments):
        start_time = i * segment_length  # 현재 세그먼트의 시작 시간 (초 단위)

        # 마지막 세그먼트의 지속 시간을 조정합니다.
        if i == num_segments - 1:
            duration = video_duration - start_time  # 남은 시간을 지속 시간으로 설정
        else:
            duration = segment_length  # 세그먼트 길이를 지속 시간으로 설정

        all_durations.append(duration)  # 세그먼트의 지속 시간을 저장

        print(f"샘플링 중인 세그먼트 {i+1}/{num_segments} (시작 시간: {int(start_time // 60)}분 {int(start_time % 60)}초)")

        # 해당 세그먼트에서 프레임을 추출합니다.
        clip = download_and_sample_video_local(
            video_path, start_time=start_time, duration=duration, frame_interval=frame_interval
        )

        # 추출된 프레임이 있으면 리스트에 추가합니다.
        if clip is not None and len(clip) > 0:
            all_segments_frames.append(clip)
        else:
            print(f"세그먼트 {i+1}에서 프레임을 추출할 수 없습니다.")

    print(f"총 {len(all_segments_frames)}개의 세그먼트가 추출되었습니다.")

    all_problematic_frames = []
    all_feedbacks = []

    # 모든 세그먼트에 대해 프레임 분석을 수행합니다.
    for idx, (segment_frames, duration) in enumerate(zip(all_segments_frames, all_durations)):
        print(f"\nAnalyzing segment {idx+1}/{len(all_segments_frames)}")
        # 각 세그먼트의 프레임들을 분석합니다.
        problematic_frames, feedbacks = analyze_frames(
            segment_frames, idx, duration, segment_length, SYSTEM_INSTRUCTION, frame_interval=frame_interval
        )
        all_problematic_frames.extend(problematic_frames)
        all_feedbacks.extend(feedbacks)

    # 문제 있는 프레임들을 시각화합니다.
    plot_problematic_frames(all_problematic_frames, all_feedbacks)

if __name__ == "__main__":
    main()
