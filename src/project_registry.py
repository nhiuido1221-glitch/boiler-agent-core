"""
project_registry.py
---------------------
Quản lý ánh xạ Telegram group -> project_id trong Supabase, phục vụ
Multi-tenant Dual-RAG: mỗi dự án (nhà máy/lò hơi mới) có kho tài liệu riêng,
tách biệt với kho tài liệu dùng chung (SHARED_PROJECT_ID).

Một Telegram group ứng với một dự án. Admin dùng lệnh /new_project trong
group đó để gán group_id hiện tại vào project_id mới hoặc đã có sẵn.

Lưu ý kỹ thuật: dùng select-rồi-insert/update thủ công thay vì upsert(on_conflict=...)
- một số phiên bản supabase-py/postgrest-py xử lý on_conflict không nhất quán
và có thể trả lỗi PGRST125 "Invalid path". Cách này chậm hơn 1 round-trip
nhưng ổn định trên mọi phiên bản client.

Nguyên tắc công nghiệp: mọi lỗi kết nối Supabase đều fail-soft (trả về giá
trị mặc định), không được làm crash luồng xử lý chính của bot.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("project_registry")

TABLE_NAME = os.getenv("SUPABASE_PROJECTS_TABLE", "boiler_projects")
SHARED_PROJECT_ID = os.getenv("SHARED_KNOWLEDGE_TAG", "shared")
DEFAULT_PROJECT_ID = os.getenv("DEFAULT_PROJECT_ID", "boiler_default")


@lru_cache(maxsize=1)
def _get_supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY chưa được cấu hình trong .env")
    # SUPABASE_URL phải đúng dạng "https://<project-ref>.supabase.co" - KHÔNG có dấu
    # "/" ở cuối, KHÔNG kèm "/rest/v1". Dán sai định dạng là nguyên nhân phổ biến
    # nhất gây lỗi PGRST125 "Invalid path specified in request URL".
    return create_client(url.rstrip("/"), key)


def register_project(project_id: str, project_name: str, group_id: str, created_by: str) -> bool:
    """
    Gán group_id hiện tại (group Telegram) vào 1 project_id. Nếu group_id đã
    từng được gán trước đó, cập nhật đè - 1 group chỉ thuộc đúng 1 project
    tại một thời điểm.
    """
    try:
        client = _get_supabase_client()
        payload = {
            "project_id": project_id,
            "project_name": project_name,
            "group_id": group_id,
            "created_by": created_by,
        }

        existing = (
            client.table(TABLE_NAME).select("group_id").eq("group_id", group_id).limit(1).execute()
        )
        if existing.data:
            client.table(TABLE_NAME).update(payload).eq("group_id", group_id).execute()
        else:
            client.table(TABLE_NAME).insert(payload).execute()

        logger.info("Đã đăng ký project_id='%s' cho group_id='%s'", project_id, group_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không đăng ký được dự án vào Supabase: %s", exc)
        return False


def get_project_id_for_group(group_id: str) -> str:
    """
    Trả về project_id đã gán cho group_id này. Nếu group chưa từng chạy
    /new_project, hoặc Supabase lỗi, fallback về DEFAULT_PROJECT_ID - không
    được làm fail luồng xử lý chính.
    """
    try:
        client = _get_supabase_client()
        response = (
            client.table(TABLE_NAME).select("project_id").eq("group_id", group_id).limit(1).execute()
        )
        if response.data:
            return response.data[0]["project_id"]
        return DEFAULT_PROJECT_ID
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Không tra cứu được dự án cho group_id=%s (dùng mặc định '%s'): %s",
            group_id,
            DEFAULT_PROJECT_ID,
            exc,
        )
        return DEFAULT_PROJECT_ID


def get_project_info(group_id: str) -> Optional[dict]:
    """Lấy đầy đủ thông tin dự án đã gán cho group_id (dùng cho lệnh /my_project)."""
    try:
        client = _get_supabase_client()
        response = client.table(TABLE_NAME).select("*").eq("group_id", group_id).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không lấy được thông tin dự án cho group_id=%s: %s", group_id, exc)
        return None


def set_boiler_type(group_id: str, boiler_type: str) -> bool:
    """
    Ghi loại thiết bị cụ thể (vd: 'Lò hơi ống lửa 10 tấn/h đốt trấu', 'Lò ghi bậc thang',
    'Lò dầu tải nhiệt Q=3.000.000 Kcal/h') cho group hiện tại. Dùng để AI trả lời cụ thể
    theo đúng loại thiết bị của dự án, thay vì nói chung chung "hệ thống lò hơi/lò dầu tải
    nhiệt". Yêu cầu group_id đã được /new_project gán trước đó.
    """
    try:
        client = _get_supabase_client()
        existing = (
            client.table(TABLE_NAME).select("group_id").eq("group_id", group_id).limit(1).execute()
        )
        if not existing.data:
            logger.warning("set_boiler_type: group_id=%s chưa có dự án, chạy /new_project trước.", group_id)
            return False
        client.table(TABLE_NAME).update({"boiler_type": boiler_type}).eq("group_id", group_id).execute()
        logger.info("Đã cập nhật boiler_type='%s' cho group_id='%s'", boiler_type, group_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không ghi được boiler_type vào Supabase: %s", exc)
        return False


def get_boiler_type_for_group(group_id: str) -> str:
    """Trả về loại thiết bị đã khai báo cho group, hoặc chuỗi rỗng nếu chưa khai báo/lỗi."""
    try:
        client = _get_supabase_client()
        response = (
            client.table(TABLE_NAME).select("boiler_type").eq("group_id", group_id).limit(1).execute()
        )
        if response.data and response.data[0].get("boiler_type"):
            return response.data[0]["boiler_type"]
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không tra cứu được boiler_type cho group_id=%s: %s", group_id, exc)
        return ""


def get_project_group_ids(project_id: str) -> list[str]:
    """
    Liệt kê TẤT CẢ group_id đang gán vào project_id này - thường chỉ 1 nhóm, nhưng
    schema cho phép nhiều nhóm Telegram khác nhau cùng gán vào 1 project_id, nên phải
    liệt kê đủ trước khi xoá dự án (không được bỏ sót nhóm nào).
    """
    try:
        client = _get_supabase_client()
        response = client.table(TABLE_NAME).select("group_id").eq("project_id", project_id).execute()
        return [row["group_id"] for row in (response.data or [])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không liệt kê được group_id cho project_id=%s: %s", project_id, exc)
        return []


def delete_project_mapping(project_id: str) -> int:
    """
    Xoá TOÀN BỘ bản ghi Supabase (mọi group_id) đang gán vào project_id này. Dùng cho
    lệnh /delete_project (Telegram, chỉ ADMIN). KHÔNG đụng tới dữ liệu Qdrant - xem
    src/rag_retriever.py delete_project_chunks() cho phần đó, gọi riêng từ nơi khác.

    Trả về số bản ghi đã xoá (0 nếu lỗi hoặc không có gì để xoá).
    """
    try:
        client = _get_supabase_client()
        response = client.table(TABLE_NAME).delete().eq("project_id", project_id).execute()
        deleted_count = len(response.data or [])
        logger.critical("Đã xoá %s bản ghi Supabase (group_id) gán với project_id='%s'.", deleted_count, project_id)
        return deleted_count
    except Exception as exc:  # noqa: BLE001
        logger.error("Lỗi xoá bản ghi Supabase cho project_id=%s: %s", project_id, exc)
        return 0
