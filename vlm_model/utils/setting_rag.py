# utils/setting_rag.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import logging

# 로깅 설정 (선택 사항)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def setup_rag():
    """
    RAG를 위한 FAISS 인덱스와 SentenceTransformer 모델을 초기화합니다.
    
    Returns:
    - model: SentenceTransformer 모델
    - index: FAISS 인덱스
    - documents: 문서 리스트
    """
    try:
        # 문제 행동 데이터 로드
        behaviors_path = Path('problematic_behaviors.json')  # 실제 경로로 수정하세요.
        if not behaviors_path.exists():
            raise FileNotFoundError(f"문제 행동 데이터 파일을 찾을 수 없습니다: {behaviors_path}")

        with behaviors_path.open("r", encoding='utf-8') as f:
            behaviors_data = json.load(f)  # behaviors_data를 올바르게 할당

        # 문서 리스트 생성 (각 문제 행동의 설명을 하나의 문서로 간주)
        documents = [
            f"{item['category']} - {item['behavior']}: {item['improvement']} {item['recommendations']}"
            for item in behaviors_data
        ]

        # SentenceTransformer 모델 로드 (CPU만 사용)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

        # 문서 임베딩 생성
        document_embeddings = model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
        document_embeddings = np.array(document_embeddings).astype('float32')

        # FAISS 인덱스 생성 (L2 거리 기반)
        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(document_embeddings)

        logger.info("RAG 컴포넌트가 성공적으로 초기화되었습니다.")
        return model, index, documents

    except Exception as e:
        logger.error(f"RAG 컴포넌트 초기화 중 오류 발생: {e}")
        raise e  # 예외를 다시 던져 애플리케이션 시작을 중단하도록 함