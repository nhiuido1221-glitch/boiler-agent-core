"""
scripts/seed_knowledge_example.py
------------------------------------
Ví dụ MINH HOẠ cách nạp tài liệu vào Qdrant collection "knowledge", gắn
project_id = SHARED_PROJECT_ID (kho DÙNG CHUNG, mọi dự án đều tham khảo được).
Đây là dữ liệu mẫu - Kỹ sư Long thay thế bằng tài liệu SOP / thông số kỹ
thuật thật của nhà máy (hoặc dùng lệnh upload tài liệu qua Telegram, xem
PHASE4_HUONG_DAN.md / PHASE5_HUONG_DAN.md).

Chạy: python scripts/seed_knowledge_example.py
(Yêu cầu đã chạy setup_qdrant_collections.py trước đó)
"""
from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()


SAMPLE_DOCUMENTS = [
    "Áp suất vận hành bình thường của lò hơi ống lửa ống khói (fire-tube boiler) "
    "thường nằm trong khoảng 7-10 bar. Nếu áp suất vượt quá mức cài đặt của van an "
    "toàn (safety valve), phải lập tức kiểm tra và xả áp theo quy trình SOP-BLR-003.",
    "Khi phát hiện rò rỉ hơi nước tại mặt bích (flange) đường ống chính, phải đóng "
    "van chặn (isolation valve) gần nhất, báo cáo kỹ sư trực và không được tự ý sửa "
    "chữa khi hệ thống còn áp suất.",
    "Nhiệt độ khói thải (flue gas) vượt quá 250 độ C so với thiết kế là dấu hiệu cần "
    "phải vệ sinh bề mặt trao đổi nhiệt (heat exchanger surface), giảm hiệu suất và "
    "tăng tiêu hao nhiên liệu.",
    "Mức nước trong lồng lò (drum level) phải duy trì trong dải an toàn hiển thị trên "
    "ống thủy (gauge glass). Mức nước thấp báo động (low water alarm) phải được xử lý "
    "ngay lập tức để tránh cháy khô (dry firing) gây nổ lò.",
]


def main() -> int:
    # Thêm sys.path để import được src.rag_retriever khi chạy script này trực tiếp
    # (python scripts/seed_knowledge_example.py) từ thư mục gốc dự án.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from src.rag_retriever import embed_texts

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")
    shared_tag = os.getenv("SHARED_KNOWLEDGE_TAG", "shared")

    if not url:
        print("LỖI: QDRANT_URL chưa được cấu hình trong .env")
        return 1

    client = QdrantClient(url=url, api_key=api_key, timeout=30)
    # task_type="RETRIEVAL_DOCUMENT" vì đây là nội dung được lưu trữ để tìm kiếm sau này.
    vectors = embed_texts(SAMPLE_DOCUMENTS, task_type="RETRIEVAL_DOCUMENT")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": doc,
                "source": "SOP_mau",
                "project_id": shared_tag,
                "incident_name": "",
                "boiler_type_tag": "chung",
            },
        )
        for doc, vector in zip(SAMPLE_DOCUMENTS, vectors)
    ]

    client.upsert(collection_name=collection, points=points)
    print(f"Đã nạp {len(points)} tài liệu mẫu vào collection '{collection}' (project_id='{shared_tag}').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
