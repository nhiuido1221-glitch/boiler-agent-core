"""
semantic_chunker.py
---------------------
Chia nhỏ văn bản theo NGỮ NGHĨA (Semantic Chunking) thay vì theo số từ cố định.

Ý TƯỞNG: Thay vì cắt cứng mỗi 200 từ (có thể cắt ngang giữa 1 ý đang trình bày
dở), ta:
  1. Tách văn bản thành từng câu.
  2. Gộp mỗi câu với 1 câu trước + 1 câu sau thành "cửa sổ" (window) để có đủ
     ngữ cảnh khi tạo embedding (câu đơn lẻ thường quá ngắn để embedding phản
     ánh đúng ý nghĩa).
  3. Tạo embedding cho tất cả cửa sổ CÙNG LÚC (1 lần gọi API Gemini theo batch).
  4. Đo "khoảng cách ngữ nghĩa" (1 - cosine similarity) giữa 2 cửa sổ liền kề.
  5. Nơi nào khoảng cách nhảy vọt (vượt ngưỡng phân vị - percentile threshold,
     tự thích nghi theo từng tài liệu) nghĩa là ở đó đang CHUYỂN Ý - cắt chunk
     tại đó. Nơi nào khoảng cách nhỏ (câu sau tiếp nối ý câu trước) thì gộp
     chung 1 chunk.

Kỹ thuật này phổ biến trong các pipeline "Advanced RAG" (tương đương
SemanticChunker của LangChain/LlamaIndex), nhưng ở đây tự cài đặt bằng chính
Gemini embedding đã tích hợp sẵn - KHÔNG cần thêm model nặng chạy tại chỗ,
tránh lặp lại lỗi tràn RAM (OOM) từng gặp trên máy chủ Render 512MB.

An toàn: nếu 1 "chunk ngữ nghĩa" vẫn bị quá dài (vd đoạn văn dài không có ranh
giới ý rõ ràng), có lớp cắt bổ sung theo số từ (_hard_split) làm lưới an toàn,
tránh 1 chunk quá lớn làm loãng vector embedding hoặc vượt giới hạn context.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("semantic_chunker")

# Cửa sổ ngữ cảnh: gộp mỗi câu với N câu trước/sau khi tạo embedding so sánh,
# giúp embedding phản ánh đúng ý nghĩa hơn là chỉ 1 câu đơn lẻ (thường quá ngắn).
_WINDOW_RADIUS = 1
# Percentile: nếu khoảng cách ngữ nghĩa giữa 2 câu liền kề nằm trong top
# (100 - PERCENTILE)% cao nhất của TOÀN VĂN BẢN, coi đó là điểm chuyển ý - cắt
# chunk tại đó. 95 là giá trị khuyến nghị phổ biến (không cắt quá vụn).
_BREAKPOINT_PERCENTILE = 95
# Lưới an toàn: 1 chunk ngữ nghĩa không được vượt quá số từ này, kể cả khi
# không tìm thấy điểm chuyển ý rõ ràng (tránh 1 chunk quá dài).
_MAX_CHUNK_WORDS = 350
# 1 chunk ngữ nghĩa quá ngắn (dưới số từ này) sẽ được gộp thêm vào chunk kế
# tiếp thay vì đứng riêng - tránh sinh ra hàng loạt chunk vụn vặt vô nghĩa.
_MIN_CHUNK_WORDS = 40

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!…:])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Tách văn bản thành câu - dùng regex đơn giản (dựa vào dấu câu kết thúc
    ".", "?", "!", ":" hoặc xuống dòng) thay vì thư viện NLP nặng, vì tài liệu
    kỹ thuật tiếng Việt ở đây chủ yếu là câu ngắn, rõ ràng, không cần tokenizer
    phức tạp."""
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return raw


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


def _hard_split(text: str, max_words: int = _MAX_CHUNK_WORDS) -> list[str]:
    """Lưới an toàn: cắt thô theo số từ CHỈ áp dụng cho 1 chunk ngữ nghĩa đã
    bị gộp quá dài - không phải phương pháp chính (khác với cách làm cũ)."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def semantic_chunk(text: str, embed_fn) -> list[str]:
    """
    Chia `text` thành các chunk theo ranh giới ngữ nghĩa.

    embed_fn: hàm nhận list[str] trả về list[list[float]] (dùng
    src.rag_retriever.embed_texts với task_type phù hợp cho việc SO SÁNH nội
    bộ văn bản, không phải để truy hồi - gọi với task_type="SEMANTIC_SIMILARITY").

    Nếu văn bản quá ngắn (< 3 câu) hoặc lỗi khi gọi API embedding (mất mạng,
    hết quota...), tự động rơi về cắt thô theo số từ (_hard_split) - đảm bảo
    KHÔNG BAO GIỜ làm hỏng toàn bộ luồng nạp tài liệu chỉ vì 1 bước tối ưu.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return _hard_split(text)

    windows = []
    for i in range(len(sentences)):
        start = max(0, i - _WINDOW_RADIUS)
        end = min(len(sentences), i + _WINDOW_RADIUS + 1)
        windows.append(" ".join(sentences[start:end]))

    try:
        vectors = embed_fn(windows)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Lỗi tạo embedding để chia chunk ngữ nghĩa (rơi về cắt theo số từ): %s", exc
        )
        return _hard_split(text)

    distances = [
        _cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]
    if not distances:
        return _hard_split(text)

    sorted_distances = sorted(distances)
    idx = min(int(len(sorted_distances) * _BREAKPOINT_PERCENTILE / 100), len(sorted_distances) - 1)
    threshold = sorted_distances[idx]

    # Gộp câu thành chunk: cắt SAU câu i nếu khoảng cách đến câu i+1 >= ngưỡng.
    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist >= threshold and len(" ".join(current).split()) >= _MIN_CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))

    # Lưới an toàn cuối: chunk nào vượt quá _MAX_CHUNK_WORDS thì cắt thêm.
    final_chunks: list[str] = []
    for chunk in chunks:
        final_chunks.extend(_hard_split(chunk))

    logger.info(
        "Semantic chunking: %s câu -> %s chunk (ngưỡng khoảng cách=%.3f).",
        len(sentences),
        len(final_chunks),
        threshold,
    )
    return final_chunks
