# utils/retrieve_feedback.py

# RAG의 Retriever 역할을 하는 함수
def retrieve_relevant_feedback(query_embedding, index, documents, top_k=5, similarity_threshold=0.7):
    """
    FAISS 인덱스를 사용하여 가장 유사한 문서를 검색합니다.
    
    Parameters:
    - query_embedding: 검색할 쿼리의 임베딩
    - index: FAISS 인덱스
    - documents: 문서 리스트
    - top_k: 검색할 상위 k개 문서
    - similarity_threshold: 유사도 임계값 (0~1)
    
    Returns:
    - List of relevant documents
    """
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    relevant_documents = []
    for distance, idx in zip(D[0], I[0]):
        similarity = 1 / (1 + distance)  # L2 거리 기반 유사도 변환
        if similarity >= similarity_threshold:
            relevant_documents.append(documents[idx])
    return relevant_documents