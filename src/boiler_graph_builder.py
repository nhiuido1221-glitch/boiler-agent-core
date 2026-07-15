"""
boiler_graph_builder.py
------------------------
Lõi kiến trúc LangGraph của hệ thống "AI Agent Lò Hơi Trung Tâm".

Luồng định tuyến:

    START
      -> entry_node              (chuẩn hoá input, tăng loop_counter)
      -> loop_guard_edge          (conditional: loop_counter >= MAX -> circuit_breaker_node;
                                    ADMIN luôn được bỏ qua Loop Guard - xem "Admin God Mode")
      -> vision_analysis_node     (nếu có ảnh đính kèm -> gọi Vision Model phân tích)
      -> query_router_node        (Task 32: phân loại nhẹ "numeric_lookup" vs "general" để
                                    điều chỉnh cách RAG truy hồi bên dưới)
      -> rag_retrieval_node       (Task 30/33: Hybrid Search (dense Gemini + sparse BM25, alpha
                                    weighting) lấy pool ứng viên RỘNG từ Qdrant knowledge + history,
                                    lọc theo project_id - CHƯA rerank)
      -> rerank_node               (Task 31/33: gọi Cohere Rerank API chọn lại top-K liên quan
                                    nhất trong pool, dựng context_text/sources cuối cùng - node
                                    riêng, có 2 lớp try/except + timeout 20s, không bao giờ treo bot)
      -> emergency_router_node    (quét từ khóa khẩn cấp trên raw_message + vision_summary - MẶC
                                    ĐỊNH TẮT từ 2026-07-11 qua EMERGENCY_KEYWORD_GATE_ENABLED=false,
                                    vì khớp chuỗi con thô gây báo động giả trên câu hỏi thông tin
                                    thông thường; bật lại bằng 1 biến môi trường khi cần)
      -> post_rule_layer_edge     (conditional: emergency > standard)
           -> emergency_handler_node  (ưu tiên tuyệt đối, phản hồi tức thì không qua LLM)
           -> standard_llm_node       (Groq LLM + Dual-RAG context, áp dụng cho MỌI role)
      -> logging_node             (ghi log Supabase, không bao giờ làm fail request chính)
      -> END

VỀ "ADMIN GOD MODE" (đã sửa lỗi thiết kế):
Bản đầu tiên cho ADMIN đi vào 1 nhánh riêng trả về câu trả lời CỐ ĐỊNH, hoàn
toàn KHÔNG gọi LLM/RAG - hậu quả là khi Kỹ sư Long (ADMIN) hỏi, hệ thống
không bao giờ thực sự phân tích dữ liệu đã upload, chỉ trả về 1 câu "đã xử lý
với quyền admin" vô nghĩa. Đây là lỗi thiết kế, không phải hành vi mong muốn.

Từ bản này, ADMIN đi qua ĐÚNG luồng standard_llm_node như mọi người (được trả
lời thật, dựa trên RAG + LLM). "God Mode" giờ nghĩa là ADMIN được đặc quyền
CAO HƠN chứ không phải bị chặn không cho hỏi:
  1) Bỏ qua Loop Guard hoàn toàn (không bao giờ bị circuit-breaker cắt mạch).
  2) System prompt của LLM được điều chỉnh: cung cấp thông tin chi tiết, kỹ
     thuật, không rào trước đón sau kiểu "nên hỏi kỹ sư khác" (vì chính người
     hỏi đã là kỹ sư trưởng).
Emergency vẫn LUÔN ưu tiên tuyệt đối, kể cả với ADMIN - an toàn con người
không bao giờ được phép bị "God Mode" ghi đè.

Nguyên tắc công nghiệp tuân thủ:
- Mọi thao tác gọi mạng (LLM call, Qdrant, Supabase) đều bọc trong try/except,
  có retry + sleep cho LLM, và "fail-soft" (không crash) cho RAG/logging.
- Không có code GUI, chạy headless, phù hợp server/container.
- Emergency là đường đi NHANH, KHÔNG gọi LLM - đảm bảo phản hồi tức thì cho
  tình huống an toàn quan trọng, không phụ thuộc độ trễ / uptime của Groq API.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.schema import AgentState

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("boiler_graph_builder")

# ==============================================================================
# CẤU HÌNH RULE LAYER
# ==============================================================================
MAX_LOOP_COUNT = int(os.getenv("MAX_LOOP_COUNT", "3"))

# Task (phản hồi anh Long 2026-07-11): cơ chế quét từ khóa đang quá "ngáo" - khớp
# CHUỖI CON thô, không phân biệt được câu hỏi thông tin ("tại sao lò bị MẤT NƯỚC")
# với báo cáo sự cố thật ("lò đang MẤT NƯỚC, cần xử lý gấp"), nên cứ thấy từ khóa
# là bắn cảnh báo khẩn cấp - gây phiền, làm giảm lòng tin vào cảnh báo thật. Theo
# yêu cầu, TẮT hẳn lớp chặn cứng này (route thẳng qua standard_llm_node, LLM vẫn
# có đủ ngữ cảnh RAG + system prompt để tự nhắc nhở an toàn khi cần, chỉ là không
# còn kiểu chặn cứng bằng khớp từ khóa). Giữ nguyên toàn bộ code/luồng graph phía
# dưới để khi cần bật lại (có thể nâng cấp thêm điều kiện ngữ cảnh, không chỉ
# khớp từ khóa thô), chỉ cần đổi biến môi trường này, KHÔNG cần sửa code.
EMERGENCY_KEYWORD_GATE_ENABLED = os.getenv("EMERGENCY_KEYWORD_GATE_ENABLED", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

EMERGENCY_KEYWORDS = [
    "tut ap",
    "tụt áp",
    "ro ri",
    "rò rỉ",
    "no",
    "nổ",
    "chay",
    "cháy",
    "qua ap",
    "quá áp",
    "mat nuoc",
    "mất nước",
    "canh bao do",
    "cảnh báo đỏ",
    "tat lua",
    "tắt lửa",
]

# Cum tu CO DINH, quet RIENG tren vision_summary (mo ta anh do chinh AI phan tich
# sinh ra) - HOAN TOAN DOC LAP voi EMERGENCY_KEYWORD_GATE_ENABLED o tren. Ly do
# tach rieng (phan hoi anh Long 2026-07-14): bao dong gia truoc day la do quet tho
# tren CAU HOI VAN BAN cua nguoi dung (vi du "tai sao lo bi MAT NUOC" - cau hoi
# thong tin, khong phai su co that) - rui ro nay KHONG ap dung cho vision_summary,
# vi day la MO TA THUC TE do chinh Vision Model quan sat tu anh SCADA, khong phai
# cau hoi gia dinh. Vi vay co the BAT LUON, khong can cho bien moi truong, ma khong
# tai dien van de bao dong gia tren cau hoi thong tin da gap truoc do.
#
# Cum tu duoc THIET KE DE VISION MODEL TU DUNG DUNG NGUYEN VAN (xem system prompt
# moi trong vision_analysis_node) khi phat hien o mau do/vang tren man hinh SCADA
# (quy uoc co dinh: nen do + chu "H" = vuot nguong cao; nen vang + chu "L" = duoi
# nguong an toan) - nho vay khop chinh xac, giam toi da rui ro bao dong gia so voi
# quet tu khoa tho tren van ban tu do nguoi dung go.
VISION_ALERT_MARKERS = [
    # Cum tu MOI (tu 2026-07-14, theo phan hoi anh Long: "do/vang" khong phai
    # thuat ngu chuyen nganh - dung dung "vuot nguong" cho canh bao do (H, qua
    # cao) va "thap" cho canh bao vang (L, duoi nguong an toan)).
    "phat hien thong so vuot nguong",
    "phát hiện thông số vượt ngưỡng",
    "phat hien thong so thap",
    "phát hiện thông số thấp",
    # Cum tu THE HE CU (truoc 2026-07-14) - giu lai lam luoi an toan phong khi
    # Vision Model vi ly do nao do (cache, phien ban khac) van tra ve cum cu.
    "phat hien canh bao do",
    "phát hiện cảnh báo đỏ",
    "phat hien canh bao vang",
    "phát hiện cảnh báo vàng",
]

MAX_RETRY = 3
RETRY_SLEEP_SECONDS = 2
MAX_IMAGES_PER_REQUEST = 5  # giới hạn của Groq vision model


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


# ==============================================================================
# NODES
# ==============================================================================
def entry_node(state: AgentState) -> AgentState:
    """
    Cổng vào (Entry Node): chuẩn hoá state đầu vào, khởi tạo giá trị mặc định,
    tăng loop_counter để Loop Guard có thể đánh giá ở bước tiếp theo.
    """
    try:
        loop_counter = state.get("loop_counter", 0) + 1
        log_line = f"[entry_node] group_id={state.get('group_id')} loop_counter={loop_counter}"
        logger.info(log_line)
        return {
            "loop_counter": loop_counter,
            "user_role": state.get("user_role", "GUEST"),
            "is_emergency": False,
            "keywords_found": [],
            "vision_summary": "",
            "rag_context": "",
            "rag_sources": [],
            "routing_log": [log_line],
        }
    except Exception as exc:  # noqa: BLE001 - không được để crash hệ thống SCADA
        logger.exception("entry_node lỗi không mong muốn: %s", exc)
        return {
            "loop_counter": state.get("loop_counter", 0) + 1,
            "routing_log": [f"[entry_node] LỖI: {exc}"],
        }


def circuit_breaker_node(state: AgentState) -> AgentState:
    """
    Loop Guard kích hoạt: cắt mạch khi loop_counter >= MAX_LOOP_COUNT
    để tránh vòng lặp vô hạn / đốt token / spam Telegram. ADMIN được miễn trừ
    (xem loop_guard_edge) nên node này chỉ chạy tới với OPERATOR/GUEST.
    """
    msg = (
        f"[circuit_breaker_node] Vòng lặp vượt ngưỡng ({state.get('loop_counter')} "
        f">= {MAX_LOOP_COUNT}). Đã cắt mạch để bảo vệ hệ thống."
    )
    logger.warning(msg)
    return {
        "final_response": (
            "Hệ thống đã phát hiện vòng lặp xử lý vượt quá ngưỡng an toàn và tự động "
            "dừng lại. Vui lòng liên hệ kỹ sư vận hành để kiểm tra trực tiếp."
        ),
        "routing_log": [msg],
    }


def vision_analysis_node(state: AgentState) -> AgentState:
    """
    Nếu tin nhắn có đính kèm ảnh, gọi Groq Vision Model (Llama-4-Scout) để mô tả
    hiện trạng thiết bị trong ảnh. Kết quả được lưu vào 'vision_summary' và sẽ
    được gộp chung với raw_message khi Emergency Router quét từ khóa (ví dụ:
    Vision Model mô tả "phát hiện vết rò rỉ dầu" -> tự động kích hoạt khẩn cấp)
    và khi standard_llm_node sinh câu trả lời.

    Nếu không có ảnh, bỏ qua ngay, không tốn chi phí gọi API.
    """
    images = state.get("images") or []
    if not images:
        return {"vision_summary": "", "routing_log": ["[vision_analysis_node] Không có ảnh đính kèm, bỏ qua."]}

    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    images_to_send = images[:MAX_IMAGES_PER_REQUEST]
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            vision_llm = ChatGroq(
                model=os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2,
                timeout=45,
            )
            content: list[dict] = [
                {
                    "type": "text",
                    "text": (
                        "Bạn là kỹ sư phân tích hình ảnh thiết bị lò hơi / lò dầu tải nhiệt "
                        "công nghiệp. Mô tả ngắn gọn những gì quan sát được trong (các) ảnh sau, "
                        "đặc biệt chú ý các dấu hiệu bất thường: rò rỉ, ăn mòn, cháy nổ, đồng hồ "
                        "áp suất/nhiệt độ bất thường, kết cấu bị hư hỏng.\n\n"
                        "LƯU Ý QUAN TRỌNG VỀ BẢNG CÀI ĐẶT GHI (Cấp liệu 1, Ghi 2, Ghi 3, Ghi 4, Ghi 5 - "
                        "có thể hiển thị dưới nhãn \"BEN 1\" đến \"BEN 5\"): đây là bảng CÀI ĐẶT PAUSE "
                        "TIME (thời gian dừng của từng tầng ghi, đơn vị giây) - các ô PV/SV/H/L/LL "
                        "trong bảng này là THÔNG SỐ ĐIỀU KHIỂN TỐC ĐỘ GHI, KHÔNG PHẢI cảm biến áp "
                        "suất/nhiệt độ/lưu lượng. TUYỆT ĐỐI KHÔNG được gọi giá trị PV trong bảng này "
                        "là \"áp suất dầu\", \"Mpa\", \"Bar\" hay bất kỳ đơn vị đo lường vật lý nào - đây "
                        "là lỗi nghiêm trọng đã từng xảy ra thực tế (PV=108 của \"Ben 5\" - vốn chỉ là "
                        "giá trị đếm Pause Time đang chạy bình thường - bị gọi nhầm thành \"áp suất dầu "
                        "108 Mpa\" và báo động giả). KHÔNG coi các bảng Cấp liệu(1)/Ghi 2-5/BEN 1-5 này "
                        "là nguồn phát hiện cảnh báo đỏ/vàng - bảng này chỉ để tham khảo cấu hình vận "
                        "hành, KHÔNG dùng để kích hoạt cảnh báo khẩn cấp.\n\n"
                        "NẾU ảnh là màn hình SCADA (HMI): hệ thống có QUY ƯỚC MÀU CỐ ĐỊNH cho từng "
                        "ô thông số - nền màu ĐỎ kèm chữ \"H\" nghĩa là thông số đó đang VƯỢT NGƯỠNG "
                        "(quá cao); nền màu VÀNG kèm chữ \"L\" nghĩa là đang THẤP HƠN NGƯỠNG AN TOÀN "
                        "(quá thấp); nền màu XANH LÁ là bình thường. Hãy QUÉT TOÀN BỘ ảnh tìm CÁC Ô "
                        "ĐỎ/VÀNG này.\n"
                        "Nếu tìm thấy BẤT KỲ ô nào nền đỏ (chữ H), BẮT BUỘC bắt đầu câu trả lời đúng "
                        "nguyên văn cụm \"Phát hiện thông số vượt ngưỡng:\" rồi liệt kê TỪNG thông số "
                        "theo đúng mẫu \"<tên thông số đọc trên màn hình bằng tiếng Việt> đang Cao "
                        "(kèm giá trị đọc được)\", ví dụ \"Mức nước bồn đang Cao (96.6%)\". "
                        "Nếu tìm thấy ô nền vàng (chữ L) mà KHÔNG có ô đỏ nào, bắt đầu đúng nguyên "
                        "văn cụm \"Phát hiện thông số thấp:\" rồi liệt kê TỪNG thông số theo đúng mẫu "
                        "\"<tên thông số đọc trên màn hình bằng tiếng Việt> đang Thấp (kèm giá trị đọc "
                        "được)\", ví dụ \"Nhiệt độ buồng đốt đang Thấp (104 độ C)\". LUÔN ưu tiên dùng "
                        "tên thông số bằng tiếng Việt hiển thị trên màn hình (không chỉ nêu mã tag kỹ "
                        "thuật như PT04, TT09 một mình) và KHÔNG dùng cụm \"cảnh báo đỏ\"/\"cảnh báo "
                        "vàng\" (không phải thuật ngữ vận hành đúng chuẩn). "
                        "CẤM TUYỆT ĐỐI TỰ SUY ĐOÁN HAY TỰ DỊCH Ý NGHĨA THÔNG SỐ: CHỈ được dùng ĐÚNG "
                        "NGUYÊN VĂN chữ nhãn THẬT SỰ NHÌN THẤY ngay tại/cạnh/phía trên ô đó trên màn "
                        "hình. TUYỆT ĐỐI KHÔNG được tự gán một cái tên nghe có vẻ hợp lý (ví dụ tự "
                        "gọi là \"áp suất\", \"tỷ lệ oxy khí thải\", \"mức nước\"...) nếu chữ đó KHÔNG "
                        "thực sự in sẵn ngay tại ô đang xét - đây là lỗi nghiêm trọng đã từng xảy ra "
                        "(cùng 1 mã tag nhưng mỗi lần bị gọi 1 tên khác nhau, có lần bị gán nhầm hẳn "
                        "thành thông số khác), khiến kỹ sư hiểu sai hoàn toàn đang xử lý thông số nào. "
                        "Nếu KHÔNG thấy nhãn tiếng Việt rõ ràng ngay tại ô đó, BẮT BUỘC dùng đúng mã "
                        "tag kỹ thuật hiển thị trên màn hình (ví dụ PT04, TT09) kèm chú thích \"(chưa "
                        "rõ tên tiếng Việt)\" ngay sau - KHÔNG được bỏ trống, càng không được bịa tên. "
                        "QUAN TRỌNG: khi ĐÃ phát hiện cảnh báo (đỏ hoặc vàng), CHỈ trả lời đúng cụm cố "
                        "định + danh sách thông số bất thường theo đúng mẫu trên - DỪNG LẠI Ở ĐÓ, "
                        "KHÔNG mô tả thêm các thông số khác đang bình thường, KHÔNG thêm đoạn tổng "
                        "quan/kết luận, KHÔNG giải thích ý nghĩa thông số - chỉ cần kết quả cảnh báo "
                        "ngắn gọn để gửi thẳng làm tin cảnh báo khẩn cấp. Nếu toàn bộ đều xanh/bình "
                        "thường (không có cảnh báo nào), KHÔNG dùng 2 cụm cố định trên, mô tả bình "
                        "thường như yêu cầu ở đầu. "
                        "Trả lời bằng tiếng Việt có dấu."
                    ),
                }
            ]
            for img_url in images_to_send:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

            response = vision_llm.invoke([HumanMessage(content=content)])
            description = getattr(response, "content", str(response))
            logger.info("[vision_analysis_node] thành công ở lần thử %s", attempt)
            return {
                "vision_summary": description,
                "routing_log": [
                    f"[vision_analysis_node] OK attempt={attempt} so_anh={len(images_to_send)}"
                ],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[vision_analysis_node] Lỗi mạng/API lần %s/%s: %s. Ngủ đông %ss rồi thử lại.",
                attempt,
                MAX_RETRY,
                exc,
                RETRY_SLEEP_SECONDS,
            )
            time.sleep(RETRY_SLEEP_SECONDS)

    error_msg = f"[vision_analysis_node] Thất bại sau {MAX_RETRY} lần thử: {last_error}"
    logger.error(error_msg)
    return {"vision_summary": "", "routing_log": [error_msg]}


# Tu khoa/tin hieu cho thay cau hoi dang TRA CUU SO LIEU/BANG BIEU cu the (Task 32:
# dinh tuyen nhe, khong goi them 1 luot LLM rieng de tranh ton do tre/chi phi - dung
# heuristic tu khoa + regex, du dung cho quy mo 1 nha may; neu sau nay thay chua du
# chinh xac, co the nang cap thanh 1 lenh goi LLM phan loai rieng).
_NUMERIC_LOOKUP_KEYWORDS = (
    "bảng", "thông số", "áp suất", "nhiệt độ", "pause", "định mức", "ngưỡng",
    "chỉ số", "tần suất", "công suất", "lưu lượng", "tỷ lệ", "%", "bar", "độ c",
    "kg/h", "m3", "phút", "giây", "giờ",
)
_NUMBER_RE = re.compile(r"\d")


def query_router_node(state: AgentState) -> AgentState:
    """
    Phan loai nhe (Task 32 - Agentic Routing) truoc khi vao rag_retrieval_node:
    cau hoi TRA SO LIEU/BANG BIEU ("numeric_lookup") can pool ung vien rong hon va
    uu tien khop tu khoa (BM25) hon la hieu ngu nghia thuan tuy, vi du lieu dang
    nam rai rac o nhieu dong bang khac nhau va sai lech 1 con so la khong chap
    nhan duoc trong van hanh cong nghiep. Cau hoi con lai ("general" - khai niem/
    quy trinh/nguyen nhan-giai phap) giu nguyen ty le hybrid mac dinh.

    CHU Y THIET KE: day la phan loai bang heuristic (tu khoa + regex so), KHONG
    goi them LLM - dung 0 chi phi/do tre, phu hop quy mo bot noi bo 1 nha may.
    Day la danh doi da duoc thong bao: neu do chinh xac phan loai chua du tot,
    huong nang cap tiep theo la 1 lenh goi LLM nho (nhu "viet lai cau hoi" da de
    xuat truoc do) de phan loai chinh xac hon, doi lay them ~0.3-0.5s do tre.
    """
    raw_message = state.get("raw_message", "")
    text_lower = _normalize(raw_message)

    has_keyword = any(kw in text_lower for kw in _NUMERIC_LOOKUP_KEYWORDS)
    has_number = bool(_NUMBER_RE.search(raw_message))
    query_type = "numeric_lookup" if (has_keyword and has_number) else "general"

    log_line = f"[query_router_node] query_type={query_type}"
    logger.info(log_line)
    return {"query_type": query_type, "routing_log": [log_line]}


def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    Hybrid Search (Task 30/33): truy hồi Qdrant (knowledge base + incident history)
    bằng dense (Gemini) + sparse (BM25) kết hợp, CHỈ trong phạm vi project_id của
    request hiện tại (+ kho dùng chung), dựa trên nội dung tin nhắn (gộp cả kết quả
    phân tích ảnh nếu có). Trả về 1 pool ỨNG VIÊN RỘNG, CHƯA rerank - bước rerank
    tách riêng thành rerank_node (Task 33 - anh Long yêu cầu tách Node riêng để dễ
    theo dõi/tắt riêng nếu Cohere API có sự cố).

    Nếu Qdrant/embedding lỗi/chưa cấu hình: trả về danh sách ứng viên rỗng - không
    làm fail request chính (fail-soft), rerank_node/standard_llm_node vẫn chạy tiếp
    bình thường, chỉ là không có ngữ cảnh RAG.
    """
    raw_message = state.get("raw_message", "")
    vision_summary = state.get("vision_summary", "")
    project_id = state.get("project_id", "")
    combined_query = f"{raw_message} {vision_summary}".strip()

    if not combined_query:
        return {
            "rag_candidates": [],
            "routing_log": ["[rag_retrieval_node] Không có nội dung để truy hồi."],
        }

    try:
        from src.rag_retriever import retrieve_candidates

        query_type = state.get("query_type", "general")
        candidates = retrieve_candidates(combined_query, project_id=project_id, query_type=query_type)
        log_line = f"[rag_retrieval_node] project_id={project_id} pool={len(candidates)} ứng viên (chưa rerank)"
        logger.info(log_line)
        return {
            "rag_candidates": candidates,
            "routing_log": [log_line],
        }
    except Exception as exc:  # noqa: BLE001 - RAG là tầng bổ trợ, không được làm fail request
        logger.warning("[rag_retrieval_node] Lỗi Hybrid Search, tiếp tục không có ứng viên: %s", exc)
        return {
            "rag_candidates": [],
            "routing_log": [f"[rag_retrieval_node] LỖI (bỏ qua): {exc}"],
        }


def rerank_node(state: AgentState) -> AgentState:
    """
    Rerank (Task 31/33 - BẮT BUỘC theo yêu cầu anh Long): nhận pool ứng viên rộng từ
    rag_retrieval_node, gọi Cohere Rerank API (module src/reranker.py) để chọn lại
    top_k liên quan nhất, rồi dựng context_text/sources cuối cùng cho standard_llm_node.

    Node RIÊNG (không gộp vào rag_retrieval_node) để: (1) dễ log/theo dõi/đo thời
    gian riêng bước rerank, (2) nếu Cohere API timeout/lỗi, chỉ node này bị ảnh
    hưởng - rag_retrieval_node đã lấy xong ứng viên từ Qdrant, không phải gọi lại.

    BẢO VỆ CHỐNG TREO BOT: rerank_candidates() nội bộ đã bọc httpx.Client(timeout=20)
    + try/except (rơi về giữ nguyên điểm hybrid nếu lỗi). Ở ĐÂY bọc thêm 1 lớp
    try/except NGOÀI CÙNG nữa (phòng lỗi import/dữ liệu bất thường ngoài phạm vi
    gọi API) - đảm bảo TUYỆT ĐỐI không có tình huống nào khiến node này crash và
    làm treo toàn bộ luồng xử lý tin nhắn Telegram.
    """
    candidates = state.get("rag_candidates", [])
    if not candidates:
        return {
            "rag_context": "",
            "rag_sources": [],
            "routing_log": ["[rerank_node] Không có ứng viên để rerank."],
        }

    raw_message = state.get("raw_message", "")
    vision_summary = state.get("vision_summary", "")
    combined_query = f"{raw_message} {vision_summary}".strip()
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    try:
        from src.reranker import rerank_candidates
        from src.rag_retriever import build_rag_context

        all_hits = rerank_candidates(combined_query, candidates, top_n=top_k)
        result = build_rag_context(all_hits)
        log_line = f"[rerank_node] {len(candidates)} ứng viên -> {len(all_hits)} sau rerank"
        logger.info(log_line)
        return {
            "rag_context": result["context_text"],
            "rag_sources": result["sources"],
            "routing_log": [log_line],
        }
    except Exception as exc:  # noqa: BLE001 - rerank là tầng bổ trợ, KHÔNG được treo/fail bot
        logger.warning(
            "[rerank_node] Lỗi rerank (rơi về dùng nguyên pool ứng viên theo điểm hybrid, "
            "không sập luồng): %s", exc,
        )
        try:
            from src.rag_retriever import build_rag_context

            fallback_hits = sorted(candidates, key=lambda h: h.get("score", 0.0), reverse=True)[:top_k]
            result = build_rag_context(fallback_hits)
            return {
                "rag_context": result["context_text"],
                "rag_sources": result["sources"],
                "routing_log": [f"[rerank_node] LỖI rerank, dùng fallback hybrid: {exc}"],
            }
        except Exception as exc2:  # noqa: BLE001 - lớp bảo vệ cuối cùng, tuyệt đối không crash
            logger.error("[rerank_node] Lỗi cả fallback (bỏ qua RAG hoàn toàn): %s", exc2)
            return {
                "rag_context": "",
                "rag_sources": [],
                "routing_log": [f"[rerank_node] LỖI cả fallback: {exc2}"],
            }


# ==============================================================================
# LOC NGUONG VAT LY HOP LY (sanity check - 2026-07-15, phan hoi anh Long sau khi
# thay 1 canh bao that co gia tri PHI THUC TE: "Ben 5 ap suat dau dang Cao (108
# Mpa)" - qua cao so voi he thuy luc cong nghiep thuc te, nhieu kha nang Vision
# Model doc nham chu so tu anh chup). Day KHONG phai nguong H/L that cua SCADA
# (SCADA da tu quyet dinh mau do/vang dung theo nguong that cua no roi) - bang
# nay CHI la luoi an toan chan cac gia tri KHONG THE nao dung trong thuc te,
# giam rui ro bao dong gia do AI doc sai anh (lech dau phay, nham don vi...).
#
# QUAN TRONG: day la GIA TRI UOC LUONG BAN DAU, dat RONG de tranh bo sot canh
# bao that - anh Long CAN chinh lai cho khop dung thong so ky thuat thuc te
# cua tung thiet bi (ap suat dinh muc, dong dinh muc dong co...) neu thay chua
# phu hop. Moi phan tu: (chuoi con de nhan dien ten thong so - khong phan biet
# hoa/thuong/dau, gia tri toi thieu hop ly, gia tri toi da hop ly).
PARAM_PLAUSIBLE_RANGES: list[tuple[str, float, float]] = [
    ("áp suất hơi", 0, 25),          # Bar - ap suat hoi lo hoi cong nghiep pho bien
    ("mức nước", 0, 100),             # % - muc nuoc luon trong khoang 0-100%
    ("pt0", 0, 100),                  # % - cac tag PT0x hien dang % (vd PT04)
    ("quạt gió cấp 1", 0, 50),        # Hz - tan so bien tan pho bien 0-50Hz
    ("quạt gió cấp 2", 0, 50),        # Hz
    ("quạt hút khói", 0, 50),         # Hz
    ("dòng", 0, 200),                 # A - dong dien dong co (ap dung chung moi dong "... dong")
    ("áp suất dầu", 0, 20),           # Mpa - he thuy luc cong nghiep pho bien toi ~20 Mpa
]


def _is_plausible_value(name: str, value: float) -> bool:
    """Kiem tra 1 gia tri doc duoc co nam trong khoang HOP LY VAT LY khong, dua
    tren PARAM_PLAUSIBLE_RANGES (khop theo chuoi con trong ten, khong phan biet
    hoa/thuong/dau). Neu KHONG co quy tac nao khop ten thong so, MAC DINH coi
    la hop le (khong loc) - tranh bo sot canh bao that chi vi thieu du lieu
    ngoung cho thong so do."""
    name_norm = _normalize(name)
    for pattern, lo, hi in PARAM_PLAUSIBLE_RANGES:
        if pattern in name_norm:
            return lo <= value <= hi
    return True


# Khop 1 dong phat hien dang "- <ten> dang Cao/Thap (<gia tri><don vi>)" do
# Vision Model sinh ra (dung mau cau da quy dinh trong vision_analysis_node).
_FINDING_LINE_RE = re.compile(
    r"^-\s*(?P<name>.+?)\s+đang\s+(?P<status>Cao|Thấp)\s*\(\s*(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>[^)]*)\)\s*$",
    re.IGNORECASE,
)


def _filter_implausible_findings(vision_summary: str) -> tuple[str, list[str]]:
    """Duyet tung dong trong vision_summary, LOAI BO cac dong khop mau phat
    hien nhung co gia tri PHI THUC TE (ngoai PARAM_PLAUSIBLE_RANGES) - cac dong
    khac (dong tieu de "Phat hien thong so...", dong khong khop mau, hoac mo ta
    binh thuong) GIU NGUYEN khong dong cham. Tra ve (van ban da loc, danh sach
    dong bi loai kem ly do - de ghi log canh bao)."""
    kept_lines: list[str] = []
    dropped_lines: list[str] = []
    for line in vision_summary.splitlines():
        m = _FINDING_LINE_RE.match(line.strip())
        if not m:
            kept_lines.append(line)
            continue
        try:
            value = float(m.group("value").replace(",", "."))
        except ValueError:
            kept_lines.append(line)
            continue
        if _is_plausible_value(m.group("name"), value):
            kept_lines.append(line)
        else:
            dropped_lines.append(line.strip())
    return "\n".join(kept_lines), dropped_lines


def emergency_router_node(state: AgentState) -> AgentState:
    """
    Rule Layer - Emergency Router: gồm 2 lớp quét ĐỘC LẬP nhau (phân tách theo
    yêu cầu anh Long 2026-07-14, sau khi xác nhận báo động giả trước đây chỉ
    xảy ra với câu hỏi văn bản, không phải mô tả ảnh):

    LỚP 1 - quét vision_summary theo VISION_ALERT_MARKERS: LUÔN LUÔN BẬT, không
    phụ thuộc EMERGENCY_KEYWORD_GATE_ENABLED. An toàn vì vision_summary là mô tả
    THỰC TẾ do chính Vision Model quan sát ảnh SCADA sinh ra (không phải câu hỏi
    giả định của người dùng), và cụm từ được thiết kế để Vision Model CHỈ dùng
    đúng nguyên văn khi thực sự thấy ô đỏ/vàng theo quy ước màu cố định của màn
    hình SCADA (xem system prompt trong vision_analysis_node) - giảm tối đa rủi
    ro khớp nhầm so với quét chuỗi con thô. TỪ 2026-07-15: sau khi khớp marker,
    LỌC THÊM theo PARAM_PLAUSIBLE_RANGES - nếu TOÀN BỘ giá trị phát hiện đều
    phi thực tế (nhiều khả năng AI đọc nhầm ảnh), KHÔNG tính là khẩn cấp.

    LỚP 2 - quét raw_message + vision_summary theo EMERGENCY_KEYWORDS (hành vi
    CŨ, giữ nguyên 100%): CHỈ BẬT khi EMERGENCY_KEYWORD_GATE_ENABLED=true (mặc
    định vẫn TẮT như anh Long đã quyết định, vì đây là lớp có rủi ro báo động
    giả trên câu hỏi thông tin đã xác nhận trước đó).

    Áp dụng cho MỌI role, kể cả ADMIN - an toàn không có ngoại lệ.
    """
    vision_summary = state.get("vision_summary", "")
    vision_text_norm = _normalize(vision_summary)
    vision_found = [kw for kw in VISION_ALERT_MARKERS if _normalize(kw) in vision_text_norm]

    if vision_found:
        filtered_summary, dropped_lines = _filter_implausible_findings(vision_summary)
        if dropped_lines:
            logger.warning(
                "[emergency_router_node] Da loc bo %d dong PHI THUC TE (gia tri vuot khoang "
                "hop ly vat ly - nhieu kha nang AI doc nham anh, xem PARAM_PLAUSIBLE_RANGES): %s",
                len(dropped_lines), dropped_lines,
            )
        remaining_findings = [
            ln for ln in filtered_summary.splitlines() if _FINDING_LINE_RE.match(ln.strip())
        ]
        if remaining_findings:
            log_line = (
                f"[emergency_router_node] Vision Model tu phat hien canh bao mau tren anh SCADA "
                f"(LOP 1 - luon bat, doc lap EMERGENCY_KEYWORD_GATE_ENABLED), con lai "
                f"{len(remaining_findings)} dong hop le sau loc: {vision_found}"
            )
            logger.warning(log_line)
            return {
                "is_emergency": True,
                "keywords_found": vision_found,
                "vision_summary": filtered_summary,
                "routing_log": [log_line],
            }
        else:
            logger.warning(
                "[emergency_router_node] Vision Model bao canh bao nhung TOAN BO gia tri deu "
                "PHI THUC TE sau khi loc (nhieu kha nang doc nham anh) - KHONG tinh la khan "
                "cap, tiep tuc kiem tra Lop 2 / standard_llm_node nhu binh thuong."
            )
            # KHONG return o day - roi xuong duoi tiep tuc nhu khong co canh bao vision nao.

    if not EMERGENCY_KEYWORD_GATE_ENABLED:
        log_line = "[emergency_router_node] Khong co canh bao mau tu vision (Lop 1). Lop chan tu khoa van ban (Lop 2) dang TAT (EMERGENCY_KEYWORD_GATE_ENABLED=false) - bo qua, di tiep standard_llm_node."
        logger.info(log_line)
        return {
            "is_emergency": False,
            "keywords_found": [],
            "routing_log": [log_line],
        }

    combined_text = _normalize(state.get("raw_message", "") + " " + vision_summary)
    found = [kw for kw in EMERGENCY_KEYWORDS if _normalize(kw) in combined_text]
    is_emergency = len(found) > 0

    log_line = f"[emergency_router_node] Lop 2 (tu khoa van ban): is_emergency={is_emergency} keywords_found={found}"
    logger.info(log_line)

    return {
        "is_emergency": is_emergency,
        "keywords_found": found,
        "routing_log": [log_line],
    }


def emergency_handler_node(state: AgentState) -> AgentState:
    """
    Xử lý sự cố khẩn cấp: ưu tiên tuyệt đối, KHÔNG gọi LLM (tránh độ trễ / phụ
    thuộc uptime Groq API trong tình huống an toàn), trả về cảnh báo ngay lập
    tức kèm từ khóa đã phát hiện.

    Tu ban nay (theo yeu cau anh Long 2026-07-15): CHI gui KET QUA canh bao
    (ten thong so + trang thai Cao/Thap), KHONG gui phan mo ta/y nghia chi
    tiet cua tung thong so binh thuong khac nua. vision_summary tu
    vision_analysis_node gio DA duoc thiet ke de CHI liet ke cac thong so BAT
    THUONG khi co canh bao (xem prompt rieng trong vision_analysis_node), nen
    dung TRUC TIEP lam noi dung canh bao - khong con boc them dong "[tu khoa]"
    hay tien to "Chi tiet tu phan tich anh SCADA:" gay trung lap/dai dong nua.
    """
    vision_summary = (state.get("vision_summary") or "").strip()
    keywords = state.get("keywords_found", [])

    if vision_summary:
        lines = ["⚠️ CẢNH BÁO KHẨN CẤP:", vision_summary]
    else:
        lines = [f"⚠️ CẢNH BÁO KHẨN CẤP: Phát hiện dấu hiệu sự cố [{', '.join(keywords)}]."]
    lines.append(
        "Kỹ sư vận hành vui lòng kiểm tra hiện trường ngay lập tức và thực hiện "
        "quy trình ứng phó khẩn cấp."
    )
    msg = "\n".join(lines)

    logger.critical("[emergency_handler_node] %s", msg)
    return {
        "final_response": msg,
        "routing_log": [f"[emergency_handler_node] escalated keywords={keywords}"],
    }


def _build_boiler_system_prompt(boiler_type: str, is_admin: bool) -> str:
    """
    Dựng System Prompt cho standard_llm_node.

    Mỗi khối bên dưới ép LLM vào một khuôn tư duy cụ thể:
      - "BẠN LÀ AI" ép model nhận vai kỹ sư trưởng vận hành (không phải chatbot
        chung chung) -> giọng văn, mức độ tự tin, và từ vựng chuyên ngành đúng
        ngữ cảnh nhà máy nhiệt/lò hơi.
      - "NGUYÊN TẮC CHỐNG BỊA ĐẶT" là hàng rào cứng: buộc model đối chiếu MỌI
        câu trả lời với đúng nguyên văn tài liệu RAG, cấm dùng kiến thức nền
        để "đoán" khi tài liệu không phủ tới - đây là điểm khác biệt lớn nhất
        so với prompt cũ (trước đây cho phép model tự bổ sung kiến thức chung
        kèm cảnh báo; giờ ép TỪ CHỐI hẳn bằng đúng 1 câu cố định, an toàn hơn
        cho môi trường vận hành thực tế nhưng đồng nghĩa bot sẽ từ chối nhiều
        câu hỏi khái niệm chung nếu tài liệu chưa có - đánh đổi có chủ đích).
      - "CẢNH GIÁC VỚI SỰ CỐ CÓ TRIỆU CHỨNG GIỐNG NHAU" giữ nguyên từ bản cũ,
        chống việc model gọi sai tên sự cố chỉ vì triệu chứng bề ngoài trùng.
      - "CẤU TRÚC CÂU TRẢ LỜI BẮT BUỘC" chỉ kích hoạt cho câu hỏi kiểu sự cố/
        hiện tượng bất thường (không ép câu hỏi khái niệm/tra cứu đơn giản
        vào khuôn 3 phần, tránh trả lời máy móc không cần thiết).
      - "PHONG CÁCH TRẢ LỜI" giữ nguyên lệnh cấm Markdown - bắt buộc phải giữ,
        vì Telegram hiển thị ký hiệu Markdown thô, mất nếu bỏ đi.
      - "RÀNG BUỘC AN TOÀN BỔ SUNG" là lưới an toàn cuối: cấm đoán số liệu an
        toàn quan trọng, ưu tiên khuyến nghị thận trọng khi thiếu căn cứ.
    """
    base_prompt = """BẠN LÀ AI:
Bạn là Kỹ sư trưởng vận hành lò hơi (steam boiler) và lò dầu tải nhiệt (thermal
oil heater) công nghiệp, giàu kinh nghiệm thực chiến qua nhiều năm vận hành và
xử lý sự cố tại nhà máy. Tác phong chuyên nghiệp, bình tĩnh, không hoảng loạn
dù tình huống khẩn cấp; dùng đúng thuật ngữ chuyên ngành nhiệt (áp suất định
mức, sinh hơi, đóng cặn, gia nhiệt, xả đáy, tụt áp, quá nhiệt, an toàn liên
động...) thay vì diễn đạt chung chung, mơ hồ.

NGUYÊN TẮC CHỐNG BỊA ĐẶT (TUYỆT ĐỐI - ưu tiên cao nhất, không có ngoại lệ):
Chỉ được phép trả lời DỰA TRÊN DUY NHẤT thông tin trong phần "TÀI LIỆU THAM
KHẢO NỘI BỘ" được cung cấp bên dưới câu hỏi (nếu có). KHÔNG được tự suy diễn,
KHÔNG được dùng kiến thức chuyên môn chung để "đoán" hay "bổ sung" khi tài
liệu không đề cập hoặc không đủ căn cứ để kết luận chắc chắn. Nếu tài liệu
KHÔNG nói đến trường hợp đang hỏi, hoặc thông tin không đủ rõ để kết luận,
BẮT BUỘC trả lời ĐÚNG NGUYÊN VĂN câu sau, không thêm/bớt/diễn giải khác:

"Dữ liệu kỹ thuật hiện tại chưa đề cập đến trường hợp này, vui lòng kiểm tra
lại thực tế hoặc liên hệ Kỹ sư trưởng."

Nếu tài liệu có đoạn liên quan, PHẢI trích dẫn theo số thứ tự (ví dụ "theo Tài
liệu 2..."), dùng ĐÚNG số liệu/ngưỡng/quy trình như trong tài liệu, không làm
tròn, không diễn giải lại số liệu an toàn. Nếu nhiều đoạn tài liệu mâu thuẫn
nhau, PHẢI chỉ rõ mâu thuẫn đó thay vì tự ý chọn 1 đoạn và bỏ qua đoạn còn lại.

NGOẠI LỆ QUAN TRỌNG (2026-07-15 - sửa lỗi thực tế: bot từ chối trả lời cả khi
đã CÓ ảnh SCADA vừa chụp): nếu tin nhắn có kèm phần "[Phân tích hình ảnh đính
kèm]" - đây là MÔ TẢ TRỰC TIẾP do Vision Model vừa quan sát được từ ảnh SCADA
THỰC TẾ ngay tại thời điểm này (không phải kiến thức chung, không phải suy
đoán) - PHẢI dùng ngay nội dung đó để trả lời câu hỏi về tình trạng thiết bị
hiện tại (ví dụ "/scada", "tình hình lò thế nào", báo cáo định kỳ), KHÔNG được
áp dụng câu từ chối "Dữ liệu kỹ thuật hiện tại chưa đề cập..." cho phần mô tả
quan sát này dù tài liệu RAG không nhắc tới - RAG là kho kiến thức tĩnh, không
thể "biết trước" số liệu tức thời của 1 ảnh vừa chụp. Quy tắc từ chối ở trên
CHỈ áp dụng cho phần GIẢI THÍCH NGUYÊN NHÂN/HƯỚNG XỬ LÝ SỰ CỐ đi xa hơn mô tả
quan sát trực tiếp (ví dụ: nếu ảnh cho thấy thông số bất thường nhưng tài liệu
không có hướng xử lý tương ứng, được phép mô tả đúng những gì quan sát thấy
rồi mới áp dụng câu từ chối riêng cho phần "cách xử lý", không từ chối toàn bộ
câu trả lời.

CẢNH GIÁC VỚI SỰ CỐ CÓ TRIỆU CHỨNG GIỐNG NHAU nhưng NGUYÊN NHÂN/BẢN CHẤT khác
nhau (ví dụ đóng keo xỉ và sụt tường buồng đốt đều gây tiếng động lớn, rung
chấn - nhưng là 2 sự cố hoàn toàn khác nhau). Đọc kỹ để xác định ĐÚNG tên sự
cố mà tài liệu mô tả, KHÔNG suy diễn tên sự cố chỉ từ triệu chứng bề ngoài nếu
tài liệu đã nêu rõ tên chính xác.

CẤU TRÚC CÂU TRẢ LỜI BẮT BUỘC khi câu hỏi liên quan sự cố/hiện tượng bất
thường (KHÔNG áp dụng cho câu hỏi khái niệm/tra cứu thông số/quy trình chung
không phải sự cố cụ thể - những câu đó trả lời tự nhiên như văn nói, không ép
theo khuôn dưới đây):
1) Hiện tượng và Nguyên nhân dự đoán (theo đúng tài liệu tham khảo).
2) Các bước xử lý khẩn cấp - đánh số rõ ràng "1)", "2)", "3)"..., LUÔN đặt
   bước liên quan AN TOÀN CON NGƯỜI lên đầu tiên nếu có (dừng thiết bị, sơ
   tán, ngắt nguồn nhiệt) trước khi tới các bước khắc phục kỹ thuật.
3) Lưu ý/Cảnh báo an toàn - PHẢI nêu rõ nếu sự cố liên quan tới áp suất,
   nhiệt độ, hoặc cạn nước (3 nhóm rủi ro nghiêm trọng nhất khi vận hành).

PHONG CÁCH TRẢ LỜI:
- Trả lời như kỹ sư trưởng đang tư vấn trực tiếp cho đồng nghiệp: tự nhiên,
  đi thẳng trọng tâm, không lặp cấu trúc rập khuôn "Câu hỏi của bạn là..."
  mỗi lần.
- LUÔN trả lời bằng tiếng Việt có dấu đầy đủ.
- TUYỆT ĐỐI KHÔNG dùng ký hiệu định dạng Markdown (###, **, *, _, dấu
  backtick, gạch đầu dòng "-", hay bảng dạng "| ô | ô |") - kênh Telegram
  hiển thị các ký hiệu này y nguyên thành chữ thô. Khi liệt kê, dùng số thứ
  tự viết liền trong câu hoặc xuống dòng kèm số "1)", "2)"; khi cần nhấn
  mạnh thì dùng từ ngữ, không dùng **in đậm**.

RÀNG BUỘC AN TOÀN BỔ SUNG:
- Không suy đoán các thông số an toàn quan trọng (áp suất tối đa, ngưỡng
  cảnh báo, quy trình khẩn cấp) nếu không có căn cứ rõ ràng từ tài liệu.
- Nếu câu hỏi liên quan an toàn mà thông tin không đủ rõ, ưu tiên khuyến
  nghị thận trọng (dừng vận hành, kiểm tra trực tiếp hiện trường) hơn là
  đưa ra câu trả lời có thể sai."""

    if boiler_type:
        base_prompt += f"\n\nTHIẾT BỊ CỦA DỰ ÁN NÀY: {boiler_type}. Khi trả lời, nói cụ thể về loại thiết bị này (không nói chung chung \"hệ thống lò hơi/lò dầu tải nhiệt\") trừ khi câu hỏi rõ ràng mang tính phổ quát."

    if is_admin:
        base_prompt += (
            "\n\nNGƯỜI HỎI: Kỹ sư trưởng Lê Đức Long (quyền ADMIN). Cung cấp thông tin "
            "chi tiết, kỹ thuật sâu, không cần rào trước đón sau kiểu \"nên hỏi kỹ sư "
            "khác\" (vì chính người hỏi đã là kỹ sư trưởng). Câu trả lời từ chối do "
            "thiếu tài liệu (nếu áp dụng) vẫn phải dùng đúng nguyên văn quy định ở "
            "trên - kể cả khi người hỏi là admin."
        )
    else:
        base_prompt += (
            "\n\nNGƯỜI HỎI: kỹ sư/nhân viên vận hành (OPERATOR). Nếu tình huống có "
            "dấu hiệu vượt quá phạm vi tài liệu hoặc có rủi ro an toàn cao, khuyến "
            "khích báo cáo/liên hệ Kỹ sư trưởng thay vì tự xử lý một mình."
        )

    return base_prompt


def standard_llm_node(state: AgentState) -> AgentState:
    """
    Xử lý qua Groq LLM (Llama-3.3-70B / gpt-oss-120b) cho MỌI role (OPERATOR
    lẫn ADMIN), có đưa vào ngữ cảnh Dual-RAG (nếu tìm thấy) và kết quả phân
    tích ảnh (nếu có).

    Dùng ChatPromptTemplate + MessagesPlaceholder (chuẩn LangChain) thay vì
    dựng list [SystemMessage, HumanMessage] thủ công như bản cũ - lý do:
      1. Tách rời System Prompt khỏi phần dữ liệu động (input, chat_history),
         dễ audit/sửa nội dung chỉ thị mà không đụng logic ghép chuỗi.
      2. MessagesPlaceholder("chat_history") sẵn sàng nhận state["messages"]
         (đã có sẵn trong AgentState, kiểu Annotated[List[Any], operator.add])
         để nối nhiều lượt hỏi-đáp thành 1 hội thoại liên tục - đúng chuẩn
         LangGraph. LƯU Ý QUAN TRỌNG: hiện tại telegram_bot.py luôn gọi graph
         với "messages": [] cho MỖI tin nhắn mới (chưa có lớp lưu/khôi phục
         lịch sử hội thoại theo từng chat) - nghĩa là chat_history hôm nay
         luôn rỗng, code chạy đúng nhưng CHƯA có trí nhớ nhiều lượt thật sự.
         Đây là điểm nối sẵn cho việc nâng cấp sau (đọc/ghi state["messages"]
         vào Supabase theo group_id/user_id), không nằm trong phạm vi yêu
         cầu lần này nên chưa triển khai.
      3. Text-only, gọi qua API Groq (không tải model cục bộ) - không phát
         sinh chi phí RAM nào trên Render so với cách gọi list message cũ.

    Mọi lời gọi mạng đều bọc trong try/except + retry/sleep theo chuẩn công
    nghiệp, không được phép làm sập hệ thống SCADA nếu mất mạng tạm thời.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    raw_message = state.get("raw_message", "")
    vision_summary = state.get("vision_summary", "")
    rag_context = state.get("rag_context", "")
    is_admin = state.get("user_role") == "ADMIN"
    chat_history = state.get("messages") or []

    # Loại thiết bị cụ thể của dự án (khai báo qua lệnh /set_boiler_type trên
    # Telegram) - giúp AI nói cụ thể "lò ghi bậc thang của dự án", thay vì nói
    # chung chung. Fail-soft: nếu Supabase lỗi hoặc chưa khai báo, bỏ qua,
    # không chặn luồng trả lời chính.
    boiler_type = ""
    try:
        from src.project_registry import get_boiler_type_for_group
        boiler_type = get_boiler_type_for_group(state.get("group_id", ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[standard_llm_node] Không lấy được boiler_type (bỏ qua): %s", exc)

    system_prompt = _build_boiler_system_prompt(boiler_type, is_admin)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    user_content_parts = [raw_message]
    if vision_summary:
        user_content_parts.append(f"[Phân tích hình ảnh đính kèm]: {vision_summary}")
    if rag_context:
        user_content_parts.append(rag_context)
    user_content = "\n\n".join(p for p in user_content_parts if p)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            llm = ChatGroq(
                model=os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2,
                timeout=30,
            )
            chain = prompt_template | llm
            response = chain.invoke({"chat_history": chat_history, "input": user_content})
            content = getattr(response, "content", str(response))
            logger.info("[standard_llm_node] thành công ở lần thử %s (admin=%s)", attempt, is_admin)
            return {
                "final_response": content,
                "routing_log": [f"[standard_llm_node] OK attempt={attempt} admin={is_admin}"],
                # Nối lượt hỏi-đáp này vào messages (operator.add) - sẵn sàng
                # cho lớp lưu lịch sử hội thoại nếu telegram_bot.py sau này
                # đọc lại state["messages"] và truyền tiếp vào lượt kế tiếp.
                "messages": [HumanMessage(content=user_content), AIMessage(content=content)],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[standard_llm_node] Lỗi mạng/API lần %s/%s: %s. Ngủ đông %ss rồi thử lại.",
                attempt, MAX_RETRY, exc, RETRY_SLEEP_SECONDS,
            )
            time.sleep(RETRY_SLEEP_SECONDS)

    error_msg = f"[standard_llm_node] Thất bại sau {MAX_RETRY} lần thử: {last_error}"
    logger.error(error_msg)
    return {
        "final_response": (
            "Hệ thống đang gặp sự cố kết nối tới dịch vụ AI. Đã ghi nhận yêu cầu và sẽ "
            "tự động thử lại. Vui lòng liên hệ kỹ sư nếu cần xử lý gấp."
        ),
        "routing_log": [error_msg],
    }


def logging_node(state: AgentState) -> AgentState:
    """
    Ghi log tương tác vào Supabase - là bước CUỐI CÙNG trước END, chạy cho
    MỌI nhánh (kể cả circuit_breaker) để đảm bảo không bỏ sót dữ liệu audit.
    Đây là side-effect, không được phép làm fail toàn bộ request nếu Supabase
    mất kết nối (đã fail-soft bên trong supabase_logger.log_interaction).
    """
    try:
        from src.supabase_logger import log_interaction

        ok = log_interaction(
            group_id=state.get("group_id", ""),
            project_id=state.get("project_id", ""),
            user_role=state.get("user_role", ""),
            raw_message=state.get("raw_message", ""),
            final_response=state.get("final_response", ""),
            is_emergency=state.get("is_emergency", False),
            keywords_found=state.get("keywords_found", []),
            routing_log=state.get("routing_log", []),
            rag_sources=state.get("rag_sources", []),
        )
        return {"routing_log": [f"[logging_node] supabase_logged={ok}"]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[logging_node] Lỗi không mong muốn khi log (bỏ qua): %s", exc)
        return {"routing_log": [f"[logging_node] LỖI (bỏ qua): {exc}"]}


# ==============================================================================
# CONDITIONAL EDGES
# ==============================================================================
def loop_guard_edge(state: AgentState) -> Literal["circuit_breaker", "continue"]:
    """
    ADMIN (God Mode) được miễn trừ Loop Guard hoàn toàn - không bao giờ bị cắt
    mạch. OPERATOR/GUEST vẫn bị chặn nếu loop_counter >= MAX_LOOP_COUNT.
    """
    if state.get("user_role") == "ADMIN":
        return "continue"
    if state.get("loop_counter", 0) >= MAX_LOOP_COUNT:
        return "circuit_breaker"
    return "continue"


def post_rule_layer_edge(state: AgentState) -> Literal["emergency", "standard"]:
    """
    Định tuyến sau Rule Layer: Emergency luôn ưu tiên tuyệt đối (kể cả với
    ADMIN - an toàn không có ngoại lệ). Mọi trường hợp còn lại (OPERATOR lẫn
    ADMIN) đều đi qua standard_llm_node để nhận câu trả lời THẬT dựa trên RAG
    + LLM, khác với thiết kế trước đó (ADMIN bị chặn ở 1 node trả lời cố định).
    """
    if state.get("is_emergency", False):
        return "emergency"
    return "standard"


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================
def build_graph():
    """
    Xây dựng và compile LangGraph hoàn chỉnh cho Boiler Agent.
    Trả về một compiled graph (có thể .invoke() / .stream() trực tiếp).
    """
    graph = StateGraph(AgentState)

    graph.add_node("entry_node", entry_node)
    graph.add_node("circuit_breaker_node", circuit_breaker_node)
    graph.add_node("vision_analysis_node", vision_analysis_node)
    graph.add_node("query_router_node", query_router_node)
    graph.add_node("rag_retrieval_node", rag_retrieval_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_node("emergency_router_node", emergency_router_node)
    graph.add_node("emergency_handler_node", emergency_handler_node)
    graph.add_node("standard_llm_node", standard_llm_node)
    graph.add_node("logging_node", logging_node)

    graph.add_edge(START, "entry_node")

    # Loop Guard: cắt mạch nếu vượt ngưỡng (trừ ADMIN), ngược lại đi tiếp tới Vision Analysis
    graph.add_conditional_edges(
        "entry_node",
        loop_guard_edge,
        {
            "circuit_breaker": "circuit_breaker_node",
            "continue": "vision_analysis_node",
        },
    )
    graph.add_edge("circuit_breaker_node", "logging_node")

    # Vision -> Dual-RAG -> Emergency Router (chuỗi tuần tự, không điều kiện)
    graph.add_edge("vision_analysis_node", "query_router_node")
    graph.add_edge("query_router_node", "rag_retrieval_node")
    graph.add_edge("rag_retrieval_node", "rerank_node")
    graph.add_edge("rerank_node", "emergency_router_node")

    # Sau Rule Layer: Emergency ưu tiên tuyệt đối, còn lại (mọi role) -> standard_llm_node
    graph.add_conditional_edges(
        "emergency_router_node",
        post_rule_layer_edge,
        {
            "emergency": "emergency_handler_node",
            "standard": "standard_llm_node",
        },
    )
    graph.add_edge("emergency_handler_node", "logging_node")
    graph.add_edge("standard_llm_node", "logging_node")

    graph.add_edge("logging_node", END)

    return graph.compile()


if __name__ == "__main__":
    # Quick smoke-test cục bộ (không gọi API thật nếu không có GROQ_API_KEY hợp lệ)
    compiled_graph = build_graph()
    test_state: AgentState = {
        "raw_message": "Báo cáo áp suất bình thường.",
        "user_role": "OPERATOR",
        "group_id": "test_group",
        "project_id": "test_project",
        "loop_counter": 0,
        "images": [],
        "messages": [],
        "routing_log": [],
    }
    print("Graph compiled OK. Nodes:", list(compiled_graph.get_graph().nodes.keys()))
