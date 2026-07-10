"""
scripts/setup_qdrant_collections.py
-------------------------------------
Script chạy 1 lần (thủ công) để tạo 2 Qdrant collection cần cho Dual-RAG:
  - QDRANT_COLLECTION_KNOWLEDGE: tài liệu kỹ thuật / SOP lò hơi
  - QDRANT_COLLECTION_HISTORY:   lịch sử sự cố đã xử lý

Đồng thời tạo payload index trên trường 'project_id' ở cả 2 collection - bắt
buộc để Multi-tenant Dual-RAG (lọc theo dự án) chạy hiệu quả (xem
src/rag_retriever.py).

Chạy: python scripts/setup_qdrant_collections.py
(Yêu cầu đã cài dependencies và điền đủ .env)

Vector size = 384 vì dùng model embedding mặc định
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (nhẹ ~220MB,
đủ hỗ trợ tiếng Việt, tránh lỗi "bad allocation" của model multilingual-e5-large
2.24GB trên máy cấu hình thường). Nếu đổi model embedding khác có số chiều
khác, script này sẽ TỰ PHÁT HIỆN và xoá + tạo lại collection cho khớp.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

VECTOR_SIZE = 384


def _get_existing_vector_size(client, name: str) -> Optional[int]:
    """Đọc số chiều vector hiện tại của 1 collection đã tồn tại, None nếu lỗi/không xác định được."""
    try:
        info = client.get_collection(name)
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        return None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        print("LỖI: QDRANT_URL chưa được cấu hình trong .env")
        return 1

    client = QdrantClient(url=url, api_key=api_key, timeout=30)

    collections = [
        os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base"),
        os.getenv("QDRANT_COLLECTION_HISTORY", "boiler_incident_history"),
    ]

    existing = {c.name for c in client.get_collections().collections}

    for name in collections:
        if name in existing:
            current_size = _get_existing_vector_size(client, name)
            if current_size is not None and current_size != VECTOR_SIZE:
                print(
                    f"[MIGRATE] Collection '{name}' đang có vector_size={current_size}, "
                    f"khác với model embedding hiện tại ({VECTOR_SIZE}). Xoá và tạo lại "
                    f"(dữ liệu cũ trong collection này, nếu có, sẽ mất)..."
                )
                client.delete_collection(name)
                existing.discard(name)

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"[TẠO MỚI] Đã tạo collection '{name}' (vector_size={VECTOR_SIZE}, distance=COSINE).")
        else:
            print(f"[OK] Collection '{name}' đã tồn tại và đúng vector_size={VECTOR_SIZE}, bỏ qua.")

        # Payload index cho project_id - bắt buộc để Multi-tenant filter hiệu quả.
        # create_payload_index tự bỏ qua nếu index đã tồn tại (không lỗi khi chạy lại).
        try:
            client.create_payload_index(
                collection_name=name,
                field_name="project_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"[OK] Đã đảm bảo payload index 'project_id' trên collection '{name}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"[CẢNH BÁO] Không tạo được payload index cho '{name}' (có thể đã tồn tại): {exc}")

    print("Hoàn tất setup Qdrant collections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
