"""
rag_retriever.py
-----------------
Dual-RAG: truy hồi song song 2 nguồn dữ liệu trong Qdrant trước khi đưa
ngữ cảnh (context) vào LLM:
  1) Collection "knowledge" (QDRANT_COLLECTION_KNOWLEDGE) - tài liệu kỹ thuật
     lò hơi / lò dầu tải nhiệt (SOP, thông số thiết kế, quy trình vận hành).
  2) Collection "history" (QDRANT_COLLECTION_HISTORY) - lịch sử sự cố /
     incident đã từng xử lý, giúp AI trả lời dựa trên kinh nghiệm thực tế.

Multi-tenant (Cải tiến): mỗi điểm dữ liệu trong Qdrant được gắn payload
'project_id'. Khi truy hồi, chỉ lấy các điểm có project_id KHỚP với dự án
hiện tại HOẶC được gắn project_id = SHARED_PROJECT_ID (kho dùng chung) -
đảm bảo tài liệu của dự án A không lẫn sang dự án B, trong khi tài liệu
dùng chung (quy chuẩn kỹ thuật tổng quát) vẫn được mọi dự án tham khảo.

Embedding: dùng fastembed (thư viện chính chủ của Qdrant, chạy ONNX local,
KHÔNG cần gọi API ngoài, KHÔNG cần thêm API key). Model mặc định:
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (384 chiều, ~220MB,
nhẹ, tránh lỗi "bad allocation" khi load model). Hỗ trợ tiếng Việt tốt.

Nguyên tắc công nghiệp: nếu Qdrant không kết nối được, KHÔNG được làm sập
luồng xử lý chính - trả về context rỗng và log cảnh báo, để hệ thống vẫn
trả lời được (chỉ là không có RAG) thay vì crash toàn bộ request.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("rag_retriever")

EMBEDDING_MODEL_NAME = os.getenv(
    "QDRANT_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Chỉ model dòng E5 (intfloat/*-e5-*) mới cần tiền tố "query: "/"passage: ". Model
# hiện tại (MiniLM multilingual) KHÔNG dùng quy ước này - thêm vào sẽ làm giảm chất lượng.
USES_E5_PREFIX = "e5" in EMBEDDING_MODEL_NAME.lower()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
SHARED_PROJECT_ID = os.getenv("SHARED_KNOWLEDGE_TAG", "shared")


@lru_cache(maxsize=1)
def _get_embedding_model():
    """
    Khởi tạo model embedding 1 lần duy nhất (lru_cache) vì load model tốn
    vài giây - không được load lại mỗi request.
    """
    from fastembed import TextEmbedding

    logger.info(
        "Đang tải embedding model: %s (lần đầu sẽ tải file model, có thể mất vài chục giây)",
        EMBEDDING_MODEL_NAME,
    )
    return TextEmbedding(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_qdrant_client():
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL chưa được cấu hình trong .env")
    return QdrantClient(url=url, api_key=api_key, timeout=15)


def embed_query(text: str) -> list[float]:
    """
    Sinh vector embedding cho 1 câu truy vấn. Chỉ thêm tiền tố "query: " nếu đang
    dùng model dòng E5 (đúng quy ước của E5) - model mặc định hiện tại (MiniLM
    multilingual) không cần và không nên thêm tiền tố này.
    """
    model = _get_embedding_model()
    text_to_embed = f"query: {text}" if USES_E5_PREFIX else text
    vector = next(model.embed([text_to_embed]))
    return vector.tolist()


def _build_project_filter(project_id: str):
    """
    Bộ lọc Qdrant: chỉ lấy điểm dữ liệu có project_id KHỚP dự án hiện tại
    HOẶC thuộc kho dùng chung (SHARED_PROJECT_ID). 'should' trong Qdrant
    Filter tương đương OR logic.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    values_to_match = {project_id, SHARED_PROJECT_ID}
    return Filter(
        should=[FieldCondition(key="project_id", match=MatchValue(value=v)) for v in values_to_match]
    )


def _search_collection(collection_name: str, query_vector: list[float], top_k: int, project_id: str) -> list[dict]:
    """
    Truy vấn 1 collection Qdrant bằng API mới `query_points` (API `search` cũ
    đã bị loại bỏ từ qdrant-client bản mới), có áp bộ lọc project_id.
    """
    client = _get_qdrant_client()
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=_build_project_filter(project_id),
            limit=top_k,
            score_threshold=RAG_SCORE_THRESHOLD,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "source": payload.get("source", collection_name),
                    "project_id": payload.get("project_id", ""),
                    "incident_name": payload.get("incident_name", ""),
                    "boiler_type_tag": payload.get("boiler_type_tag", ""),
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không truy vấn được collection '%s': %s", collection_name, exc)
        return []


def list_documents(project_id: str) -> dict[str, dict[str, int]]:
    """
    Liệt kê các tài liệu (nguồn + số lượng chunk) đã nạp cho project_id này (+ kho dùng
    chung), bằng cách SCROLL trực tiếp Qdrant - KHÔNG qua embedding/similarity search.
    Đây là truy vấn liệt kê (metadata), khác với retrieve_dual_rag_context là truy vấn
    ngữ nghĩa - dùng cho lệnh /list_docs để trả lời chính xác câu hỏi kiểu "trong kho đã
    có tài liệu nào" mà semantic search không xử lý tốt.

    Trả về: {"knowledge": {"ten_file.pdf": so_chunk, ...}, "history": {...}}
    """
    knowledge_collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    history_collection = os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history")

    client = _get_qdrant_client()
    result: dict[str, dict[str, int]] = {}

    for label, collection_name in (("knowledge", knowledge_collection), ("history", history_collection)):
        sources: dict[str, int] = {}
        try:
            next_offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=_build_project_filter(project_id),
                    limit=200,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    src = (point.payload or {}).get("source", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                if next_offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không scroll được collection '%s': %s", collection_name, exc)
        result[label] = sources

    return result


def retrieve_dual_rag_context(query: str, project_id: str = "", top_k: int = RAG_TOP_K) -> dict:
    """
    Hàm chính: truy hồi song song cả 2 collection (knowledge + history),
    CHỈ trong phạm vi project_id hiện tại + kho dùng chung, trả về dict gồm
    text ngữ cảnh đã gộp và danh sách nguồn tham khảo.

    Nếu quá trình embedding hoặc Qdrant lỗi (mất mạng, sai API key,
    collection chưa được tạo...), trả về context rỗng thay vì raise
    exception - giữ đúng chuẩn công nghiệp "không được làm sập luồng chính".
    """
    knowledge_collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    history_collection = os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history")
    effective_project_id = project_id or os.getenv("DEFAULT_PROJECT_ID", "boiler_default")

    try:
        query_vector = embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi tạo embedding cho RAG, bỏ qua RAG cho request này: %s", exc)
        return {"context_text": "", "sources": []}

    knowledge_hits = _search_collection(knowledge_collection, query_vector, top_k, effective_project_id)
    history_hits = _search_collection(history_collection, query_vector, top_k, effective_project_id)

    all_hits = sorted(knowledge_hits + history_hits, key=lambda h: h["score"], reverse=True)

    if not all_hits:
        return {"context_text": "", "sources": []}

    # Đánh số + hiện độ liên quan (%) rõ ràng cho từng đoạn, để LLM có thể trích dẫn
    # cụ thể ("theo Tài liệu 2") và tự đánh giá mức độ tin cậy thay vì coi mọi đoạn
    # ngang nhau. Bọc trong dấu === để LLM phân biệt rạch ròi đâu là dữ liệu tham
    # khảo nội bộ, đâu là câu hỏi của người dùng.
    context_lines = []
    for idx, h in enumerate(all_hits, start=1):
        header_parts = [f"Tài liệu {idx}", f"nguồn: {h['source']}", f"độ liên quan: {h['score']:.0%}"]
        if h.get("incident_name"):
            header_parts.append(f"SỰ CỐ: {h['incident_name']}")
        if h.get("boiler_type_tag") and h["boiler_type_tag"] != "chung":
            header_parts.append(f"chỉ áp dụng: {h['boiler_type_tag']}")
        context_lines.append(f"[{' | '.join(header_parts)}]\n{h['text']}")
    context_text = (
        "=== TÀI LIỆU THAM KHẢO NỘI BỘ (ưu tiên cao nhất, trích dẫn theo số thứ tự) ===\n"
        "LƯU Ý: mỗi [Tài liệu N] chỉ nói về ĐÚNG MỘT sự cố ghi trong nhãn 'SỰ CỐ:' của nó "
        "(nếu có). KHÔNG được trộn lẫn thông tin giữa 2 tài liệu khác nhãn SỰ CỐ với nhau. "
        "Nếu 1 tài liệu ghi 'chỉ áp dụng: <loại lò>' mà khác với loại lò của dự án này, phải "
        "nói rõ giới hạn đó trước khi dùng.\n\n"
        + "\n\n".join(context_lines)
        + "\n=== HẾT TÀI LIỆU THAM KHẢO NỘI BỘ ==="
    )

    return {
        "context_text": context_text,
        "sources": [
            {
                "source": h["source"],
                "project_id": h["project_id"],
                "incident_name": h.get("incident_name", ""),
                "score": round(h["score"], 4),
            }
            for h in all_hits
        ],
    }
