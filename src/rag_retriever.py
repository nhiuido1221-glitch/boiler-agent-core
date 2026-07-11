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

Embedding: gọi API ngoài Google Gemini (model "gemini-embedding-001", REST endpoint
batchEmbedContents, 768 chiều) thay vì chạy model tại chỗ. Lý do đổi: chạy embedding
local (fastembed/onnxruntime) tốn RAM quá lớn so với giới hạn 512MB của gói máy chủ
miễn phí (Render free/Starter), từng gây crash "Ran out of memory" giữa lúc xử lý câu
hỏi. Gọi API ngoài giúp giải phóng hoàn toàn RAM đó, đổi lại cần 1 API key miễn phí của
Google (GEMINI_API_KEY, lấy tại https://aistudio.google.com/apikey, gói free: 100
request/phút, 1000 request/ngày - dư sức cho quy mô dùng nội bộ 1 nhà máy). Dùng đúng
task_type cho từng chiều: "RETRIEVAL_DOCUMENT" khi nạp tài liệu, "RETRIEVAL_QUERY" khi
tìm kiếm theo câu hỏi - giúp độ chính xác tốt hơn hẳn so với kiểu tiền tố "query:"/
"passage:" thủ công của các model cũ.

Hybrid Search (Nâng cấp "Advanced RAG lite"): thay vì chỉ tìm bằng vector (dense,
hiểu ngữ nghĩa nhưng đôi khi bỏ lỡ từ khóa/số liệu chính xác), giờ kết hợp thêm
BM25 (sparse, khớp từ khóa/số liệu chính xác - mạnh cho câu hỏi tra thông số/bảng
biểu) qua thư viện `rank_bm25` (thuần Python, không tải model, không tốn RAM đáng
kể). 2 điểm số được gộp theo trọng số RAG_HYBRID_ALPHA (0=chỉ BM25, 1=chỉ vector,
mặc định 0.5) - đúng theo yêu cầu "hybrid search + alpha weighting" của anh Long,
nhưng KHÔNG dùng model embedding nặng (BGE-m3) như đề xuất gốc vì sẽ tái lặp lỗi
OOM 512MB đã sửa - đây là quyết định anh Long đã chọn ("dùng bản thay thế qua API"
thay vì nâng cấp gói Render trả phí).

Nguyên tắc công nghiệp: nếu Qdrant không kết nối được, KHÔNG được làm sập
luồng xử lý chính - trả về context rỗng và log cảnh báo, để hệ thống vẫn
trả lời được (chỉ là không có RAG) thay vì crash toàn bộ request. Tương tự,
nếu BM25/rerank lỗi, tự động rơi về kết quả vector thuần thay vì crash.
"""
from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("rag_retriever")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
GEMINI_EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:batchEmbedContents"
)
# Gộp nhiều đoạn văn bản vào 1 lần gọi HTTP (batch) để tiết kiệm số request/ngày của
# gói free (1000 RPD) - giới hạn an toàn dưới mức tối đa không công bố của API.
GEMINI_BATCH_SIZE = 90
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
SHARED_PROJECT_ID = os.getenv("SHARED_KNOWLEDGE_TAG", "shared")

# --- Hybrid search (Task 30) ---
# Trọng số gộp điểm dense (vector) vs sparse (BM25): final = alpha*dense + (1-alpha)*sparse.
RAG_HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
# Số ứng viên lấy rộng ra trước khi rerank (Task 31) chọn lại top_k cuối cùng.
RAG_CANDIDATE_POOL_SIZE = int(os.getenv("RAG_CANDIDATE_POOL_SIZE", "20"))
# Trần số điểm scroll ra để đánh BM25 - tránh scroll toàn bộ kho nếu kho quá lớn
# (kho nội bộ 1 nhà máy hiếm khi vượt mức này; nếu vượt, BM25 sẽ chỉ tính trên
# phần đầu quét được thay vì tốn quá nhiều thời gian/băng thông).
BM25_SCROLL_CAP = int(os.getenv("RAG_BM25_SCROLL_CAP", "500"))

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tách từ đơn giản (lowercase + regex \\w+) - đủ dùng cho BM25 tiếng Việt có dấu,
    không cần stemming (BM25 chỉ cần khớp từ y hệt, không cần hiểu nghĩa)."""
    return _TOKEN_RE.findall(text.lower())


# Lỗi thực tế gặp trên production (2026-07-11): "429 Too Many Requests" khi nạp tài
# liệu - nguyên nhân là gói free Gemini giới hạn 100 request/phút, trong khi 1 lần
# nạp tài liệu có cấu trúc nhiều PHẦN/Mục sẽ gọi Gemini NHIỀU LẦN (mỗi section 1 lần
# cho semantic chunking + 1 lần cuối để embed toàn bộ chunk) - dồn dập trong vài giây
# dễ vượt ngưỡng, nhất là khi có thêm lưu lượng hỏi-đáp RAG chạy song song. Trước đây
# hàm này KHÔNG có cơ chế retry - gặp 429 là raise ngay, làm fail cả lần upload dù
# đây chỉ là giới hạn TẠM THỜI (thường tự hết sau vài giây). Đúng theo nguyên tắc công
# nghiệp bắt buộc (mọi lệnh gọi mạng phải tự retry khi lỗi tạm thời) - đã thêm
# exponential backoff retry riêng cho 429/5xx bên dưới.
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
GEMINI_RETRY_BASE_SECONDS = float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "2"))


def _post_with_retry(client, url: str, headers: dict, payload: dict):
    """
    Gọi POST tới Gemini API, tự retry với exponential backoff khi gặp lỗi TẠM THỜI:
    429 (rate limit - ưu tiên đọc header "Retry-After" nếu Gemini có trả về, chính
    xác hơn đoán mò) hoặc 5xx (lỗi phía Google, thường tự hết). KHÔNG retry với lỗi
    4xx khác (400 sai payload, 401/403 sai API key, 404 sai model...) - những lỗi này
    retry cũng vô ích, chỉ tổ chờ lâu hơn trước khi báo lỗi thật cho người dùng.
    """
    import httpx

    last_exc: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == GEMINI_MAX_RETRIES:
                    logger.error(
                        "Gemini API vẫn lỗi %s sau %s lần thử - báo lỗi thật, không thử lại nữa.",
                        response.status_code, GEMINI_MAX_RETRIES,
                    )
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else GEMINI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini API trả về %s (lần thử %s/%s) - ngủ %.1fs rồi thử lại.",
                    response.status_code, attempt, GEMINI_MAX_RETRIES, sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            return response
        except httpx.TransportError as exc:  # noqa: BLE001 - mất mạng/timeout, cũng nên retry
            last_exc = exc
            sleep_seconds = GEMINI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Lỗi mạng gọi Gemini API (lần thử %s/%s): %s - ngủ %.1fs rồi thử lại.",
                attempt, GEMINI_MAX_RETRIES, exc, sleep_seconds,
            )
            if attempt == GEMINI_MAX_RETRIES:
                raise
            time.sleep(sleep_seconds)

    if last_exc:  # pragma: no cover - phòng hờ, logic trên đã raise/return hết các nhánh
        raise last_exc
    raise RuntimeError("Gemini API: hết số lần thử lại mà không rõ nguyên nhân.")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Gọi Gemini API (batchEmbedContents) để sinh embedding cho 1 danh sách văn bản
    trong ít lượt gọi HTTP nhất có thể (gộp theo GEMINI_BATCH_SIZE). Dùng chung cho
    cả nạp tài liệu (task_type="RETRIEVAL_DOCUMENT") lẫn tìm kiếm theo câu hỏi
    (task_type="RETRIEVAL_QUERY") - Gemini tối ưu vector khác nhau cho từng vai trò
    này, chính xác hơn hẳn so với dùng chung 1 kiểu embedding cho cả 2 phía.

    gemini-embedding-001 mặc định trả 3072 chiều đã chuẩn hoá sẵn; khi dùng
    output_dimensionality nhỏ hơn (768, để nhẹ Qdrant hơn) PHẢI tự chuẩn hoá lại
    (chia cho độ dài vector) - API không tự làm việc này cho model 001 (chỉ model 002
    trở lên mới tự động).

    Tự động retry (exponential backoff) khi gặp 429/5xx/lỗi mạng tạm thời - xem
    _post_with_retry() ở trên. Chỉ raise thật khi đã thử hết GEMINI_MAX_RETRIES lần.
    """
    import httpx

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình trong .env")

    all_vectors: list[list[float]] = []
    with httpx.Client(timeout=60) as client:
        for i in range(0, len(texts), GEMINI_BATCH_SIZE):
            batch = texts[i : i + GEMINI_BATCH_SIZE]
            payload = {
                "requests": [
                    {
                        "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "output_dimensionality": GEMINI_EMBEDDING_DIM,
                    }
                    for t in batch
                ]
            }
            response = _post_with_retry(
                client,
                GEMINI_EMBED_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                payload=payload,
            )
            data = response.json()
            for emb in data.get("embeddings", []):
                values = emb.get("values", [])
                norm = sum(v * v for v in values) ** 0.5
                if norm > 0:
                    values = [v / norm for v in values]
                all_vectors.append(values)

    if len(all_vectors) != len(texts):
        raise RuntimeError(
            f"Gemini API trả về {len(all_vectors)} vector nhưng gửi đi {len(texts)} đoạn văn bản - không khớp."
        )
    return all_vectors


@lru_cache(maxsize=1)
def _get_qdrant_client():
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL chưa được cấu hình trong .env")
    return QdrantClient(url=url, api_key=api_key, timeout=15)


def embed_query(text: str) -> list[float]:
    """Sinh vector embedding cho 1 câu truy vấn (task_type="RETRIEVAL_QUERY")."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


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


def _point_to_dict(point_id, score: float, payload: dict, collection_name: str) -> dict:
    return {
        "id": str(point_id),
        "score": score,
        "text": payload.get("text", ""),
        "source": payload.get("source", collection_name),
        "project_id": payload.get("project_id", ""),
        "incident_name": payload.get("incident_name", ""),
        "boiler_type_tag": payload.get("boiler_type_tag", ""),
    }


def _search_collection(
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    project_id: str,
    score_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Truy vấn dense (vector) 1 collection Qdrant bằng API `query_points`, có áp bộ
    lọc project_id. `score_threshold=None` dùng ngưỡng mặc định RAG_SCORE_THRESHOLD;
    truyền 0.0 khi cần lấy pool rộng cho hybrid search (không cắt sớm theo ngưỡng).
    """
    client = _get_qdrant_client()
    threshold = RAG_SCORE_THRESHOLD if score_threshold is None else score_threshold
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=_build_project_filter(project_id),
            limit=top_k,
            score_threshold=threshold,
            with_payload=True,
        )
        return [
            _point_to_dict(point.id, point.score, point.payload or {}, collection_name)
            for point in response.points
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không truy vấn được collection '%s': %s", collection_name, exc)
        return []


def _scroll_all_chunks(collection_name: str, project_id: str, cap: int = BM25_SCROLL_CAP) -> list[dict]:
    """
    Quét (scroll) toàn bộ điểm dữ liệu khớp project_id trong 1 collection, dùng làm
    kho ứng viên cho BM25 (BM25 cần thấy toàn bộ tập văn bản để tính điểm khớp từ,
    khác với dense search vốn tra cứu trực tiếp qua chỉ mục vector). Có trần `cap`
    để tránh quét không giới hạn nếu kho quá lớn - đủ dùng cho quy mô 1 nhà máy.
    """
    client = _get_qdrant_client()
    chunks: list[dict] = []
    try:
        next_offset = None
        while len(chunks) < cap:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=_build_project_filter(project_id),
                limit=min(200, cap - len(chunks)),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                chunks.append(_point_to_dict(point.id, 0.0, point.payload or {}, collection_name))
            if next_offset is None or not points:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không scroll được collection '%s' để đánh BM25: %s", collection_name, exc)
    return chunks


def _bm25_search(collection_name: str, query: str, top_k: int, project_id: str) -> list[dict]:
    """
    Tìm sparse (từ khóa) bằng BM25 trên toàn bộ chunk khớp project_id của 1
    collection. Điểm số được chuẩn hoá về [0, 1] (chia cho điểm cao nhất trong
    pool) để gộp được với điểm dense (vốn đã trong khoảng tương tự) ở bước hybrid.

    Nếu rank_bm25 lỗi/import fail hoặc pool rỗng: trả về [] - hybrid search phía
    trên sẽ tự rơi về dùng thuần điểm dense, không crash.
    """
    chunks = _scroll_all_chunks(collection_name, project_id)
    if not chunks:
        return []

    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(_tokenize(query))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi tính BM25 cho collection '%s' (bỏ qua sparse search): %s", collection_name, exc)
        return []

    max_score = max(raw_scores) if len(raw_scores) else 0.0
    for chunk, raw_score in zip(chunks, raw_scores):
        chunk["score"] = (raw_score / max_score) if max_score > 0 else 0.0

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_k]


def _hybrid_search(
    collection_name: str,
    query: str,
    query_vector: list[float],
    top_k: int,
    project_id: str,
    alpha: float = RAG_HYBRID_ALPHA,
) -> list[dict]:
    """
    Kết hợp dense (vector, hiểu ngữ nghĩa) + sparse (BM25, khớp từ khóa/số liệu
    chính xác) trên CÙNG 1 collection, gộp điểm theo trọng số alpha:
        final_score = alpha * dense_norm + (1 - alpha) * sparse_norm
    alpha=1 -> chỉ dùng dense (hành vi cũ), alpha=0 -> chỉ dùng BM25.

    Cả 2 nhánh đều lấy rộng hơn top_k cuối cùng (dùng chính top_k truyền vào, vốn
    thường là RAG_CANDIDATE_POOL_SIZE khi gọi từ retrieve_dual_rag_context) để có
    đủ ứng viên tốt trước khi rerank ở bước sau (Task 31) hoặc cắt trực tiếp nếu
    rerank không khả dụng.
    """
    dense_hits = _search_collection(collection_name, query_vector, top_k, project_id, score_threshold=0.0)
    sparse_hits = _bm25_search(collection_name, query, top_k, project_id) if alpha < 1.0 else []

    # Chuẩn hoá min-max điểm dense trong pool này để cùng thang đo với sparse
    # (đã chuẩn hoá 0-1 ở _bm25_search) - tránh trường hợp cosine score co cụm
    # hẹp (vd 0.3-0.45) làm alpha weighting mất tác dụng.
    dense_scores = [h["score"] for h in dense_hits]
    d_min, d_max = (min(dense_scores), max(dense_scores)) if dense_scores else (0.0, 0.0)
    d_range = (d_max - d_min) or 1.0

    merged: dict[str, dict] = {}
    for h in dense_hits:
        dense_norm = (h["score"] - d_min) / d_range
        item = dict(h)
        item["dense_score"] = round(h["score"], 4)
        item["sparse_score"] = 0.0
        item["score"] = alpha * dense_norm
        merged[h["id"]] = item

    for h in sparse_hits:
        sparse_norm = h["score"]  # đã chuẩn hoá 0-1 sẵn
        if h["id"] in merged:
            merged[h["id"]]["sparse_score"] = round(sparse_norm, 4)
            merged[h["id"]]["score"] += (1 - alpha) * sparse_norm
        else:
            item = dict(h)
            item["dense_score"] = 0.0
            item["sparse_score"] = round(sparse_norm, 4)
            item["score"] = (1 - alpha) * sparse_norm
            merged[h["id"]] = item

    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]


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


def retrieve_candidates(
    query: str,
    project_id: str = "",
    top_k: int = RAG_TOP_K,
    query_type: str = "general",
) -> list[dict]:
    """
    Bước 1/2 của "Advanced RAG lite": Hybrid search (dense Gemini + sparse BM25,
    alpha weighting) trên cả 2 collection (knowledge + history), CHỈ trong phạm vi
    project_id hiện tại + kho dùng chung, trả về 1 pool ỨNG VIÊN RỘNG (chưa rerank).

    Tách riêng khỏi bước rerank (xem reranker.rerank_candidates(), gọi từ node
    rerank_node riêng trong boiler_graph_builder.py) để rerank là 1 LangGraph Node
    độc lập, dễ log/theo dõi/tắt riêng nếu Cohere API có sự cố, không ảnh hưởng tới
    bước truy hồi Qdrant.

    query_type ("numeric_lookup" từ query_router_node) mở rộng pool_size và ưu tiên
    BM25 hơn (alpha thấp hơn) - vì tra số liệu/bảng biểu cần khớp từ khóa/đơn vị
    chính xác hơn là "hiểu ngữ nghĩa" chung chung.

    Nếu lỗi embedding/Qdrant: trả về [] thay vì raise - giữ chuẩn công nghiệp
    "không được làm sập luồng chính".
    """
    knowledge_collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    history_collection = os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history")
    effective_project_id = project_id or os.getenv("DEFAULT_PROJECT_ID", "boiler_default")

    try:
        query_vector = embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi tạo embedding cho RAG, bỏ qua RAG cho request này: %s", exc)
        return []

    if query_type == "numeric_lookup":
        pool_size = max(RAG_CANDIDATE_POOL_SIZE, top_k) + 10
        alpha = max(0.0, RAG_HYBRID_ALPHA - 0.2)
    else:
        pool_size = max(RAG_CANDIDATE_POOL_SIZE, top_k)
        alpha = RAG_HYBRID_ALPHA

    knowledge_candidates = _hybrid_search(
        knowledge_collection, query, query_vector, pool_size, effective_project_id, alpha=alpha
    )
    history_candidates = _hybrid_search(
        history_collection, query, query_vector, pool_size, effective_project_id, alpha=alpha
    )
    return sorted(knowledge_candidates + history_candidates, key=lambda h: h["score"], reverse=True)


def build_rag_context(all_hits: list[dict]) -> dict:
    """
    Bước cuối: từ danh sách hit CUỐI CÙNG (đã rerank hoặc chưa, tuỳ rerank_node có
    chạy thành công hay không), dựng context_text đánh số + nhãn rõ ràng để đưa vào
    prompt LLM, và danh sách sources gọn để log/hiển thị.
    """
    if not all_hits:
        return {"context_text": "", "sources": []}

    # Đánh số + hiện độ liên quan (%) rõ ràng cho từng đoạn, để LLM có thể trích dẫn
    # cụ thể ("theo Tài liệu 2") và tự đánh giá mức độ tin cậy thay vì coi mọi đoạn
    # ngang nhau. Bọc trong dấu === để LLM phân biệt rạch ròi đâu là dữ liệu tham
    # khảo nội bộ, đâu là câu hỏi của người dùng.
    context_lines = []
    for idx, h in enumerate(all_hits, start=1):
        header_parts = [f"Tài liệu {idx}", f"nguồn: {h['source']}", f"độ liên quan: {max(0.0, min(1.0, h['score'])):.0%}"]
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


def retrieve_dual_rag_context(
    query: str,
    project_id: str = "",
    top_k: int = RAG_TOP_K,
    query_type: str = "general",
) -> dict:
    """
    Wrapper TƯƠNG THÍCH NGƯỢC: gộp retrieve_candidates() + rerank_candidates() +
    build_rag_context() trong 1 lệnh gọi - dùng cho script/CLI hoặc bất kỳ nơi nào
    KHÔNG chạy qua LangGraph (nơi rerank là 1 node riêng, xem rerank_node trong
    boiler_graph_builder.py). Luồng chính (Telegram bot) dùng 2 node tách riêng
    rag_retrieval_node + rerank_node, KHÔNG gọi hàm này.
    """
    from src.reranker import rerank_candidates

    candidates = retrieve_candidates(query, project_id=project_id, top_k=top_k, query_type=query_type)
    if not candidates:
        return {"context_text": "", "sources": []}

    all_hits = rerank_candidates(query, candidates, top_n=top_k)
    return build_rag_context(all_hits)


def count_project_chunks(project_id: str) -> dict[str, int]:
    """
    Đếm số chunk (KHÔNG kèm kho dùng chung, CHỈ đúng project_id này) trong cả 2
    collection - dùng để hiện số liệu cảnh báo trước khi ADMIN xác nhận xoá dự án
    qua lệnh /delete_project (Telegram). Khác với list_documents()/_build_project_filter
    (vốn OR thêm cả SHARED_PROJECT_ID cho mục đích truy hồi) - ở đây PHẢI đếm CHÍNH XÁC
    riêng project_id, không được lẫn số liệu của kho dùng chung.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    knowledge_collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    history_collection = os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history")
    client = _get_qdrant_client()
    exact_filter = Filter(must=[FieldCondition(key="project_id", match=MatchValue(value=project_id))])

    counts: dict[str, int] = {}
    for label, collection_name in (("knowledge", knowledge_collection), ("history", history_collection)):
        try:
            result = client.count(collection_name=collection_name, count_filter=exact_filter, exact=True)
            counts[label] = result.count
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không đếm được chunk collection '%s' cho project_id='%s': %s", collection_name, project_id, exc)
            counts[label] = 0
    return counts


def delete_project_chunks(project_id: str) -> dict[str, int]:
    """
    XOÁ VĨNH VIỄN toàn bộ chunk gắn ĐÚNG project_id này khỏi cả 2 collection Qdrant.
    Dùng cho lệnh /delete_project (Telegram, chỉ ADMIN, có xác nhận qua nút bấm trước
    khi gọi hàm này - xem src/telegram_bot.py).

    BẢO VỆ CỨNG: từ chối tuyệt đối nếu project_id == SHARED_PROJECT_ID - đây là kho
    dùng chung cho MỌI dự án, xoá nhầm sẽ xoá tri thức của toàn bộ nhà máy chứ không
    riêng 1 dự án. Raise ValueError thay vì âm thầm bỏ qua, để lỗi này KHÔNG BAO GIỜ
    lọt qua được kể cả khi có bug ở tầng gọi hàm.
    """
    if project_id == SHARED_PROJECT_ID:
        raise ValueError(
            f"TỪ CHỐI: '{project_id}' là kho DÙNG CHUNG, không được xoá qua lệnh xoá dự án."
        )
    if not project_id.strip():
        raise ValueError("project_id rỗng - từ chối xoá để tránh xoá nhầm toàn bộ collection.")

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    knowledge_collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    history_collection = os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history")
    client = _get_qdrant_client()
    exact_filter = Filter(must=[FieldCondition(key="project_id", match=MatchValue(value=project_id))])

    before_counts = count_project_chunks(project_id)

    deleted: dict[str, int] = {}
    for label, collection_name in (("knowledge", knowledge_collection), ("history", history_collection)):
        try:
            client.delete(collection_name=collection_name, points_selector=exact_filter)
            deleted[label] = before_counts.get(label, 0)
            logger.critical(
                "Đã XOÁ VĨNH VIỄN %s chunk của project_id='%s' trong collection '%s'.",
                deleted[label], project_id, collection_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Lỗi xoá chunk project_id='%s' trong collection '%s': %s", project_id, collection_name, exc)
            deleted[label] = 0
    return deleted
