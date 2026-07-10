"""
schema.py
---------
Dinh nghia AgentState (TypedDict) - trang thai toan cuc duoc truyen qua
tung node trong LangGraph cua he thong Boiler Agent (Multi-tenant).

Thiet ke:
- total=False: khong phai node nao cung ghi day du moi truong trong 1 luot chay.
- messages / routing_log dung Annotated[..., operator.add] de LangGraph tu dong
  CONG DON (append) thay vi ghi de moi lan node tra ve gia tri moi.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- Input tho tu nguoi dung / Telegram Bot ---
    raw_message: str
    user_role: Literal["ADMIN", "OPERATOR", "GUEST"]
    group_id: str
    project_id: str
    images: List[str]  # URL hoac base64 anh dinh kem, dung cho chan doan qua Vision Model

    # --- Ket qua phan tich anh (Phase 4: vision_analysis_node) ---
    vision_summary: str

    # --- Phan loai cau hoi (Task 32: query_router_node) - dieu chinh cach RAG
    # truy hoi: "numeric_lookup" (tra bang/thong so, uu tien khop tu khoa) vs
    # "general" (khai niem/quy trinh, uu tien hieu ngu nghia) ---
    query_type: Literal["numeric_lookup", "general"]

    # --- Ung vien tho tu Hybrid Search (Task 33: rag_retrieval_node), CHUA rerank -
    # rerank_node doc rieng field nay, KHONG doc lai Qdrant ---
    rag_candidates: List[dict]

    # --- Ket qua Dual-RAG sau rerank (Task 31/33: rerank_node) ---
    rag_context: str
    rag_sources: List[dict]

    # --- Rule Layer flags (Emergency Router) ---
    is_emergency: bool
    keywords_found: List[str]

    # --- Loop Guard (chong vong lap vo han / dot token) ---
    loop_counter: int

    # --- Lich su hoi thoai - LangGraph tu dong cong don qua operator.add ---
    messages: Annotated[List[Any], operator.add]

    # --- Nhat ky dinh tuyen, phuc vu debug / audit / truy vet cho Ky su Long ---
    routing_log: Annotated[List[str], operator.add]

    # --- Ket qua cuoi cung tra ve cho tang giao tiep (Telegram Bot) ---
    final_response: Optional[str]
