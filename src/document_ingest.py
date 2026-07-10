"""
document_ingest.py
--------------------
Trích xuất văn bản từ file (.txt, .pdf, .docx), chia nhỏ (chunk), sinh
embedding và nạp vào Qdrant collection kiến thức (knowledge). Dùng chung cho
cả lệnh upload qua Telegram (ADMIN) và các script CLI (seed dữ liệu mẫu).

CHIA THEO CẤU TRÚC TÀI LIỆU (quan trọng - tránh lẫn lộn sự cố):
Nhiều tài liệu SOP xử lý sự cố lò hơi có cấu trúc rõ ràng dạng:
    PHẦN I: CÁC SỰ CỐ HỆ THỐNG NƯỚC, HƠI VÀ ÁP LỰC
    Mục 1: Sự cố cạn nước nghiêm trọng (Mất dấu thủy)
Nếu chia nhỏ theo SỐ TỪ CỐ ĐỊNH, 1 đoạn (chunk) rất dễ chứa PHẦN CUỐI của sự
cố này lẫn PHẦN ĐẦU sự cố kế tiếp -> AI trả lời NHẦM sự cố này thành sự cố
khác (rất nguy hiểm với tài liệu an toàn). split_by_incident_sections() phát
hiện tiêu đề "PHẦN x:"/"Mục n:" để chia đúng ranh giới; nếu không có cấu
trúc này, tự rơi về chia theo số từ cố định (chunk_text).

XỬ LÝ BẢNG BIỂU TRONG PDF (quan trọng - tránh "câu ghép vô nghĩa"):
Trích xuất text thông thường (pypdf) làm PHẲNG bảng biểu thành 1 chuỗi văn
bản chảy liên tục, mất hết cấu trúc hàng/cột (vd bảng "Pause Time theo tầng
ghi" mất hẳn tiêu đề cột khi bị cắt chunk) -> LLM đọc 1 đoạn số liệu rời rạc
không có ngữ cảnh, trả lời máy móc/vô nghĩa. Hàm extract_pdf_tables() dùng
pdfplumber để nhận diện ĐÚNG cấu trúc bảng, rồi "làm phẳng có kiểm soát":
biến MỖI HÀNG thành 1 câu tự giải thích đầy đủ tên cột (vd "Hiện tượng: Lớp
liệu quá dày đầu ghi | Ghi 2: Pause ↓ | Ghi 3: Pause ↓"), để dù đứng độc lập
trong 1 chunk, câu đó vẫn có đủ ngữ cảnh để AI hiểu đúng, không bịa.

Khác với các module RAG/log khác vốn fail-soft, hàm ingest_document() ở đây
CHỦ ĐỘNG raise exception khi lỗi - vì đây là hành động chủ động của ADMIN
(upload tài liệu), cần báo lỗi cụ thể ngay cho người dùng thay vì âm thầm bỏ qua.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from io import BytesIO
from typing import Optional

logger = logging.getLogger("document_ingest")

CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "200"))
CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "70"))
SUPPORTED_EXTENSIONS = ("txt", "pdf", "docx")

_SECTION_HEADER_RE = re.compile(r"^(PHẦN\s+[IVXLCDM]+\s*:.*|Mục\s+\d+\s*:.*)$", re.MULTILINE)
_TOC_LEADER_RE = re.compile(r"\.{3,}")
_MIN_SECTION_WORDS = 15

_BOILER_TYPE_KEYWORDS = [
    "lò ghi bậc thang",
    "lò tầng sôi",
    "lò hơi ống lửa",
    "lò hơi ống nước",
    "lò dầu tải nhiệt",
    "lò ghi xích",
]


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Trích xuất văn bản thô (dạng chảy liên tục) từ file theo phần mở rộng."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        import docx

        document = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError(
        f"Định dạng file '.{ext}' chưa được hỗ trợ. Chỉ hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}."
    )


def _flatten_table(rows: list[list], table_index: int, context_hint: str = "") -> str:
    """
    Biến 1 bảng (list các hàng, mỗi hàng list các ô) thành văn bản mà MỖI HÀNG là 1 câu
    tự giải thích đầy đủ "tên cột: giá trị", để khi đứng riêng lẻ trong 1 chunk vẫn đủ
    ngữ cảnh, không cần phải "nhớ" dòng tiêu đề cột ở xa.
    """
    if not rows:
        return ""

    header = rows[0]
    title = f"BẢNG {table_index}" + (f" ({context_hint})" if context_hint else "") + ":"
    lines = [title]

    for row in rows[1:]:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue
        if len(non_empty) == 1:
            # Dòng chỉ có 1 ô có nội dung -> khả năng cao là dòng ghi chú/gộp ô, không
            # phải dòng dữ liệu đầy đủ cột - ghi thành ghi chú riêng thay vì gán nhầm tên cột.
            lines.append(f"Ghi chú: {non_empty[0]}")
            continue
        parts = []
        for idx, cell in enumerate(cells):
            if not cell:
                continue
            col_name = header[idx].strip() if idx < len(header) and header[idx] else f"cột {idx + 1}"
            parts.append(f"{col_name}: {cell}")
        lines.append(" | ".join(parts))

    return "\n".join(lines) if len(lines) > 1 else ""


def extract_pdf_tables(file_bytes: bytes) -> list[dict]:
    """
    Trích xuất RIÊNG các bảng biểu trong PDF bằng pdfplumber (nhận diện đúng cấu trúc
    hàng/cột, khác hẳn pypdf chỉ lấy text chảy liên tục). Trả về list các dict
    {"context_hint": str, "text": str} - mỗi dict là 1 bảng đã "làm phẳng có kiểm soát".
    Nếu file không có bảng hoặc pdfplumber lỗi, trả về [] (không chặn luồng ingest chính,
    vì phần text thường vẫn được trích xuất qua extract_text() như bình thường).
    """
    try:
        import pdfplumber
    except Exception as exc:  # noqa: BLE001
        logger.warning("Thư viện pdfplumber không sẵn sàng, bỏ qua trích xuất bảng: %s", exc)
        return []

    results: list[dict] = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Lỗi trích xuất bảng ở 1 trang PDF (bỏ qua trang đó): %s", exc)
                    continue
                if not tables:
                    continue

                page_text = page.extract_text() or ""
                context_hint = page_text.strip().split("\n")[0][:120] if page_text.strip() else ""

                for t_idx, table in enumerate(tables, start=1):
                    flattened = _flatten_table(table, t_idx, context_hint)
                    if flattened:
                        results.append({"context_hint": context_hint, "text": flattened})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi mở PDF bằng pdfplumber để trích xuất bảng: %s", exc)
        return []

    return results


def _detect_boiler_type_tag(heading: str) -> str:
    """Nếu tiêu đề/ngữ cảnh nhắc rõ 1 loại lò cụ thể, gắn nhãn đó; ngược lại trả về 'chung'."""
    normalized = heading.lower()
    for kw in _BOILER_TYPE_KEYWORDS:
        if kw in normalized:
            return kw
        # Một số tài liệu chỉ nhắc "ghi bậc thang" mà không kèm chữ "lò" phía trước
        # (vd tiêu đề bảng "...vận hành buồng đốt ghi bậc thang") - vẫn phải nhận diện
        # đúng loại lò thay vì rơi về "chung" một cách bỏ sót.
        short_kw = kw.replace("lò ", "", 1)
        if short_kw and short_kw != kw and short_kw in normalized:
            return kw
    return "chung"


def split_by_incident_sections(text: str) -> list[dict]:
    """
    Tách văn bản thành từng SỰ CỐ riêng biệt dựa theo tiêu đề "PHẦN x:"/"Mục n:".
    Trả về [] nếu không phát hiện được cấu trúc (< 2 tiêu đề thật, đã loại mục lục).
    """
    matches = [m for m in _SECTION_HEADER_RE.finditer(text) if not _TOC_LEADER_RE.search(m.group(0))]
    if len(matches) < 2:
        return []

    sections: list[dict] = []
    current_part = ""

    for idx, m in enumerate(matches):
        heading_line = m.group(0).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end]

        if heading_line.startswith("PHẦN"):
            current_part = heading_line
            continue

        content_words = segment.split()
        if len(content_words) < _MIN_SECTION_WORDS:
            continue

        sections.append(
            {
                "heading": heading_line,
                "part": current_part,
                "content": segment,
                "boiler_type_tag": _detect_boiler_type_tag(heading_line),
            }
        )

    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Chia văn bản thành các đoạn ~chunk_size từ, có overlap để không mất ngữ cảnh ở ranh giới.
    Dùng làm PHƯƠNG ÁN DỰ PHÒNG (fallback) cho _prose_chunk() bên dưới, và cho bảng biểu
    (bảng đã "làm phẳng có kiểm soát" theo hàng nên chia cố định là hợp lý, không cần
    semantic chunking)."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def _semantic_embed_fn(texts: list[str]) -> list[list[float]]:
    """Adapter truyền vào semantic_chunk(): dùng embedding Gemini task_type
    "SEMANTIC_SIMILARITY" (tối ưu cho so sánh mức độ giống nhau giữa 2 đoạn văn bản,
    khác với RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY dùng khi tìm kiếm)."""
    from src.rag_retriever import embed_texts

    return embed_texts(texts, task_type="SEMANTIC_SIMILARITY")


def _prose_chunk(text: str) -> list[str]:
    """
    Chia văn bản THƯỜNG (không phải bảng biểu) theo NGỮ NGHĨA (semantic chunking,
    Task 29/33) thay vì cắt cứng theo số từ - giữ đúng ranh giới ý/đoạn thay vì cắt
    giữa chừng 1 ý đang trình bày dở. Nếu Gemini API lỗi (mất mạng, hết quota...),
    tự rơi về chunk_text() cố định - KHÔNG được làm sập luồng upload tài liệu vì
    1 bước tối ưu chất lượng.
    """
    try:
        from src.semantic_chunker import semantic_chunk

        chunks = semantic_chunk(text, embed_fn=_semantic_embed_fn)
        if chunks:
            return chunks
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi semantic chunking (rơi về chia theo số từ cố định): %s", exc)
    return chunk_text(text)


def _build_chunk_records(text: str, filename: str, table_blocks: Optional[list[dict]] = None) -> list[dict]:
    """
    Xây danh sách chunk cuối cùng (kèm payload) để nạp vào Qdrant:
      1) Phần văn bản thường: ưu tiên chia theo cấu trúc sự cố (PHẦN/Mục), fallback
         chia theo số từ cố định nếu không có cấu trúc.
      2) Phần bảng biểu (nếu có, chỉ PDF): mỗi bảng là 1 nhóm chunk RIÊNG, KHÔNG bao giờ
         trộn với văn bản thường hay bảng khác - giữ nguyên các hàng đã "làm phẳng có
         kiểm soát" nên tự thân mỗi chunk đã đủ ngữ cảnh.
    """
    records: list[dict] = []

    sections = split_by_incident_sections(text)
    if sections:
        logger.info("Phát hiện %s sự cố riêng biệt trong '%s', chia theo cấu trúc.", len(sections), filename)
        for section in sections:
            for sub_chunk in _prose_chunk(section["content"]):
                labeled_text = f"[SỰ CỐ: {section['heading']}]\n{sub_chunk}"
                records.append(
                    {
                        "text": labeled_text,
                        "incident_name": section["heading"],
                        "part_name": section["part"],
                        "boiler_type_tag": section["boiler_type_tag"],
                    }
                )
    else:
        logger.info("Không phát hiện cấu trúc PHẦN/Mục trong '%s', dùng chia theo số từ cố định.", filename)
        for chunk in _prose_chunk(text):
            records.append({"text": chunk, "incident_name": "", "part_name": "", "boiler_type_tag": "chung"})

    for table in table_blocks or []:
        table_words = table["text"].split()
        # Hau het bang trong SOP van hanh khong qua dai; neu qua dai moi can chia nho tiep,
        # va luon giu dong tieu de "BANG N (...)" o dau moi chunk de khong mat ngu canh.
        table_title_line = table["text"].split("\n", 1)[0]
        table_body = table["text"].split("\n", 1)[1] if "\n" in table["text"] else ""
        sub_chunks = chunk_text(table_body, chunk_size=CHUNK_SIZE_WORDS * 2, overlap=CHUNK_OVERLAP_WORDS)
        if not sub_chunks:
            sub_chunks = [table_body]
        for sub_chunk in sub_chunks:
            records.append(
                {
                    "text": f"{table_title_line}\n{sub_chunk}",
                    "incident_name": f"Bảng: {table['context_hint']}" if table["context_hint"] else "Bảng dữ liệu",
                    "part_name": "",
                    "boiler_type_tag": _detect_boiler_type_tag(table["context_hint"]),
                }
            )

    return records


def ingest_document(
    file_bytes: bytes,
    filename: str,
    project_id: str,
    collection_name: Optional[str] = None,
) -> int:
    """
    Trích xuất (kèm bóc tách bảng biểu nếu là PDF), chia nhỏ theo cấu trúc, embed và
    nạp toàn bộ nội dung file vào Qdrant, gắn payload 'project_id' để Dual-RAG lọc đúng
    phạm vi, cùng 'incident_name' / 'boiler_type_tag' để tránh lẫn lộn giữa các sự cố.

    Trả về số lượng chunk đã nạp. Raise exception nếu thất bại - nơi gọi hàm chịu trách
    nhiệm bắt lỗi và báo cụ thể cho người dùng.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

    text = extract_text(file_bytes, filename)

    table_blocks: list[dict] = []
    if filename.lower().endswith(".pdf"):
        table_blocks = extract_pdf_tables(file_bytes)
        if table_blocks:
            logger.info("Phát hiện %s bảng biểu trong '%s', bóc tách riêng bằng pdfplumber.", len(table_blocks), filename)

    records = _build_chunk_records(text, filename, table_blocks)
    if not records:
        raise ValueError("Không trích xuất được nội dung văn bản từ file (file rỗng hoặc lỗi định dạng).")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    collection = collection_name or os.getenv("QDRANT_COLLECTION_KNOWLEDGE", "boiler_knowledge_base")

    if not url:
        raise RuntimeError("QDRANT_URL chưa được cấu hình trong .env")

    client = QdrantClient(url=url, api_key=api_key, timeout=60)
    # Dùng Gemini API để sinh embedding (KHÔNG chạy model tại chỗ nữa) - tránh hẳn lỗi
    # "Ran out of memory" trên Render free tier 512MB. task_type="RETRIEVAL_DOCUMENT" vì
    # đây là nội dung được LƯU TRỮ để tìm kiếm sau này (khác với câu hỏi truy vấn).
    from src.rag_retriever import embed_texts

    texts_to_embed = [r["text"] for r in records]
    vectors = embed_texts(texts_to_embed, task_type="RETRIEVAL_DOCUMENT")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": record["text"],
                "source": filename,
                "project_id": project_id,
                "incident_name": record["incident_name"],
                "part_name": record["part_name"],
                "boiler_type_tag": record["boiler_type_tag"],
            },
        )
        for record, vector in zip(records, vectors)
    ]

    try:
        delete_filter = Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=filename)),
                FieldCondition(key="project_id", match=MatchValue(value=project_id)),
            ]
        )
        client.delete(collection_name=collection, points_selector=delete_filter)
        logger.info("Đã xoá chunk cũ (nếu có) của file '%s' trong project_id='%s' trước khi nạp lại.", filename, project_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không xoá được chunk cũ của '%s' (có thể là lần nạp đầu tiên, bỏ qua): %s", filename, exc)

    client.upsert(collection_name=collection, points=points)
    logger.info(
        "Đã nạp %s chunk (%s bảng) từ file '%s' vào project_id='%s'",
        len(points),
        len(table_blocks),
        filename,
        project_id,
    )
    return len(points)
