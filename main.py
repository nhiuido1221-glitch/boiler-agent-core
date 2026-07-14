"""
main.py
-------
Bọc LangGraph của Boiler Agent bằng FastAPI, sẵn sàng deploy lên Render/Railway
hoặc chạy qua Docker/docker-compose local. Từ Phase 4, lúc startup sẽ tự động
khởi chạy Telegram bot (long-polling) trên background thread, dùng chung 1
compiled_graph với endpoint /invoke - đảm bảo Telegram và API trả lời nhất quán.

Endpoints:
  GET  /health            -> health check cho load balancer / Render / Railway
  POST /invoke             -> endpoint chính, gọi trực tiếp (dùng để test/tích hợp)
  GET  /graph/nodes        -> debug: liệt kê node hiện có trong graph
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("main")

from src.boiler_graph_builder import build_graph  # noqa: E402  (phải load sau load_dotenv)
from src.telegram_bot import start_telegram_bot_background, send_notification  # noqa: E402

ADMIN_ID = os.getenv("ADMIN_ID", "")

# Compile graph 1 lần duy nhất lúc khởi động app (tránh compile lại mỗi request)
try:
    compiled_graph = build_graph()
    logger.info("LangGraph đã compile thành công.")
except Exception as exc:  # noqa: BLE001 - không được để app crash lúc khởi động
    logger.exception("Lỗi compile LangGraph: %s", exc)
    compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if compiled_graph is not None:
        try:
            start_telegram_bot_background(compiled_graph)
        except Exception as exc:  # noqa: BLE001 - Telegram là kênh phụ, không được làm fail app
            logger.exception("Lỗi khởi động Telegram bot (server API vẫn chạy bình thường): %s", exc)
    else:
        logger.warning("Graph chưa sẵn sàng, bỏ qua khởi động Telegram bot.")
    yield


app = FastAPI(
    title="Boiler Agent Core API",
    description="Hệ thống AI Agent Lò Hơi Trung Tâm - Multi-tenant, Distributed Architecture",
    version="1.2.0",
    lifespan=lifespan,
)


class InvokeRequest(BaseModel):
    raw_message: str = Field(..., description="Nội dung tin nhắn thô từ người dùng")
    user_role: Literal["ADMIN", "OPERATOR", "GUEST"] = Field(default="GUEST")
    group_id: str = Field(default="unknown_group")
    project_id: str = Field(default="unknown_project")
    images: list[str] = Field(default_factory=list)
    telegram_user_id: Optional[str] = Field(
        default=None, description="ID Telegram của người gửi, dùng để xác thực ADMIN"
    )
    notify_telegram: bool = Field(
        default=False,
        description=(
            "Neu True, /invoke se chu dong day final_response (+ anh neu co) vao "
            "Telegram group_id ngay sau khi xu ly xong - dung cho cac nguon goi "
            "truc tiep (khong qua webhook Telegram goc), vi du Boiler Station "
            "Agent. Mac dinh False de tranh spam Telegram khi /invoke duoc goi de "
            "test/tich hop. Emergency (is_emergency=True) LUON duoc day di, khong "
            "phu thuoc co nay."
        ),
    )


class InvokeResponse(BaseModel):
    final_response: str
    is_emergency: bool
    routing_log: list[str]
    rag_sources: list[dict]
    processing_time_ms: int


def _resolve_role(payload: InvokeRequest) -> str:
    """
    Bảo mật: chỉ công nhận user_role == 'ADMIN' nếu telegram_user_id khớp ADMIN_ID.
    Từ chối mọi ID lạ với quyền ADMIN, hạ cấp xuống OPERATOR để an toàn.
    """
    if payload.user_role == "ADMIN":
        if not ADMIN_ID or payload.telegram_user_id != ADMIN_ID:
            logger.warning(
                "Từ chối giả mạo ADMIN: telegram_user_id=%s không khớp ADMIN_ID cấu hình.",
                payload.telegram_user_id,
            )
            return "OPERATOR"
    return payload.user_role


@app.get("/health")
@app.head("/health")  # HEAD: mot so dich vu keep-alive (UptimeRobot,
# cron-job.org...) mac dinh gui HEAD chu khong phai GET - truoc day thieu
# route nay nen bi tra ve 405 lien tuc (thay trong log that), lam sai lech
# ket qua giam sat uptime du server van chay binh thuong.
def health_check() -> dict[str, Any]:
    return {
        "status": "ok" if compiled_graph is not None else "degraded",
        "graph_ready": compiled_graph is not None,
        "telegram_enabled": os.getenv("TELEGRAM_POLL_ENABLED", "true").strip().lower()
        in ("1", "true", "yes"),
        "env": os.getenv("APP_ENV", "development"),
    }


@app.get("/graph/nodes")
def graph_nodes() -> dict[str, Any]:
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="Graph chưa được compile.")
    try:
        nodes = list(compiled_graph.get_graph().nodes.keys())
        return {"nodes": nodes}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi lấy danh sách node: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/invoke", response_model=InvokeResponse)
def invoke_agent(payload: InvokeRequest) -> InvokeResponse:
    if compiled_graph is None:
        raise HTTPException(status_code=503, detail="Graph chưa sẵn sàng, kiểm tra log khởi động.")

    start_time = time.time()
    safe_role = _resolve_role(payload)

    initial_state = {
        "raw_message": payload.raw_message,
        "user_role": safe_role,
        "group_id": payload.group_id,
        "project_id": payload.project_id,
        "images": payload.images,
        "loop_counter": 0,
        "messages": [],
        "routing_log": [],
    }

    try:
        result = compiled_graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001 - không được để request làm sập server
        logger.exception("Lỗi khi invoke graph: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {exc}") from exc

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Day Telegram khi: (a) ben goi chu dong yeu cau qua notify_telegram=True, HOAC
    # (b) Rule Layer xac dinh day la tinh huong khan cap - emergency LUON duoc day,
    # khong phu thuoc co notify_telegram cua ben goi. Boc try/except NGOAI CUNG:
    # loi gui Telegram TUYET DOI khong duoc lam fail request /invoke chinh - day
    # nguyen tac cong nghiep xuyen suot du an (station SCADA khong duoc phep sap
    # chi vi kenh thong bao phu loi tam thoi).
    should_notify = payload.notify_telegram or result.get("is_emergency", False)
    if should_notify:
        try:
            sent_ok = send_notification(
                group_id=payload.group_id,
                text=result.get("final_response", ""),
                image_data_urls=payload.images,
                is_emergency=result.get("is_emergency", False),
            )
            logger.info(
                "Da day Telegram cho request /invoke (group_id=%s, notify_telegram=%s, "
                "is_emergency=%s): %s",
                payload.group_id, payload.notify_telegram, result.get("is_emergency", False), sent_ok,
            )
        except Exception as exc:  # noqa: BLE001 - khong duoc lam fail /invoke chi vi loi Telegram
            logger.exception("Loi khi day Telegram tu /invoke (bo qua, van tra response binh thuong): %s", exc)

    return InvokeResponse(
        final_response=result.get("final_response", "Không có phản hồi."),
        is_emergency=result.get("is_emergency", False),
        routing_log=result.get("routing_log", []),
        rag_sources=result.get("rag_sources", []),
        processing_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
