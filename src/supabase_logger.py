"""
supabase_logger.py
--------------------
Ghi log moi luot xu ly (interaction) vao Supabase (PostgreSQL) de Ky su Long
tra cuu lich su, thong ke su co, phuc vu audit.

Bang can tao truoc trong Supabase (xem sql/supabase_schema.sql):
  boiler_agent_logs (
      id            bigint generated always as identity primary key,
      group_id      text,
      project_id    text,
      user_role     text,
      raw_message   text,
      final_response text,
      is_emergency  boolean,
      keywords_found text[],
      routing_log   text[],
      rag_sources   jsonb,
      created_at    timestamptz default now()
  )

Nguyen tac cong nghiep: log la tac vu phu (side-effect), KHONG duoc phep lam
that bai request chinh neu Supabase mat ket noi. Moi loi deu bat va chi ghi
canh bao vao log console, khong raise exception ra ngoai.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("supabase_logger")

TABLE_NAME = os.getenv("SUPABASE_LOG_TABLE", "boiler_agent_logs")


@lru_cache(maxsize=1)
def _get_supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY chua duoc cau hinh trong .env")
    return create_client(url, key)


def log_interaction(
    group_id: str,
    project_id: str,
    user_role: str,
    raw_message: str,
    final_response: str,
    is_emergency: bool,
    keywords_found: Optional[list[str]] = None,
    routing_log: Optional[list[str]] = None,
    rag_sources: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """
    Ghi 1 dong log vao Supabase. Tra ve True/False de node goi ham nay biet
    ket qua (chi de log/debug, khong dung de quyet dinh routing).
    """
    try:
        client = _get_supabase_client()
        payload = {
            "group_id": group_id,
            "project_id": project_id,
            "user_role": user_role,
            "raw_message": raw_message,
            "final_response": final_response,
            "is_emergency": is_emergency,
            "keywords_found": keywords_found or [],
            "routing_log": routing_log or [],
            "rag_sources": rag_sources or [],
        }
        client.table(TABLE_NAME).insert(payload).execute()
        return True
    except Exception as exc:  # noqa: BLE001 - log la side-effect, khong duoc crash request chinh
        logger.warning("Khong ghi duoc log vao Supabase (bo qua, khong anh huong request chinh): %s", exc)
        return False
