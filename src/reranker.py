"""
reranker.py
------------
Lớp Rerank (Task 31): sau khi hybrid search (rag_retriever.py) lấy 1 nhóm ứng
viên RỘNG (candidate pool), dùng Cohere Rerank API (model rerank-v3.5, đa ngôn
ngữ, hỗ trợ tiếng Việt) để xếp hạng lại chính xác hơn theo mức độ liên quan
thực sự với câu hỏi.

Vì sao cần bước này dù đã có hybrid search: cả dense (vector) lẫn sparse (BM25)
đều so sánh query với TỪNG document ĐỘC LẬP (bi-encoder / thống kê tần suất từ),
nên dễ bị "gần đúng nhưng sai trọng tâm" (vd 2 sự cố khác nhau nhưng dùng chung
nhiều từ kỹ thuật). Reranker là cross-encoder - xem xét query+document CÙNG LÚC
trong 1 model, cho điểm liên quan chính xác hơn hẳn, đây là lớp mà đề xuất gốc
của anh Long (BGE-Reranker) yêu cầu là BẮT BUỘC.

Dùng qua REST API trực tiếp (httpx) thay vì SDK chính thức `cohere` để không
thêm phụ thuộc nặng vào requirements.txt, và thay vì chạy BGE-Reranker tại chỗ
(~1-2GB RAM, sẽ tái lặp lỗi OOM 512MB) - đúng quyết định "dùng bản thay thế qua
API" anh Long đã chọn.

Nguyên tắc công nghiệp: nếu Cohere API lỗi/hết quota/mất mạng/chưa cấu hình API
key, KHÔNG được làm sập luồng chính - tự động rơi về giữ nguyên thứ tự candidates
đã có (đã được sort theo điểm hybrid từ rag_retriever.py) và chỉ cắt lấy top_n.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("reranker")

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def rerank_candidates(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    candidates: list[dict], mỗi phần tử PHẢI có key "text" (nội dung để rerank) và
    đã được sort giảm dần theo "score" (điểm hybrid) từ trước - dùng làm phương án
    dự phòng nếu rerank không khả dụng.

    Trả về: top_n phần tử của candidates, sắp xếp lại theo độ liên quan thực sự
    (Cohere "relevance_score" ghi đè vào key "score"), giữ nguyên các key khác
    (source, project_id, incident_name, boiler_type_tag...).
    """
    if not candidates:
        return []

    if not COHERE_API_KEY:
        logger.info("COHERE_API_KEY chưa cấu hình - bỏ qua bước rerank, dùng điểm hybrid gốc.")
        return candidates[:top_n]

    import httpx

    documents = [c.get("text", "") for c in candidates]
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                COHERE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": COHERE_RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi gọi Cohere Rerank API (rơi về điểm hybrid gốc, không chặn luồng chính): %s", exc)
        return candidates[:top_n]

    results = data.get("results", [])
    if not results:
        return candidates[:top_n]

    reranked: list[dict] = []
    for r in results:
        idx = r.get("index")
        if idx is None or not (0 <= idx < len(candidates)):
            continue
        item = dict(candidates[idx])
        item["score"] = r.get("relevance_score", item.get("score", 0.0))
        item["rerank_applied"] = True
        reranked.append(item)

    return reranked if reranked else candidates[:top_n]
