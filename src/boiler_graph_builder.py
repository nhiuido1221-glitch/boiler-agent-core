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
      -> rag_retrieval_node       (Dual-RAG: truy hồi Qdrant knowledge + history, lọc theo project_id)
      -> emergency_router_node    (quét từ khóa khẩn cấp trên raw_message + vision_summary)
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
                        "áp suất/nhiệt độ bất thường, kết cấu bị hư hỏng. Trả lời bằng tiếng Việt có dấu."
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


def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    Dual-RAG: truy hồi Qdrant (knowledge base + incident history), CHỈ trong
    phạm vi project_id của request hiện tại (+ kho dùng chung), dựa trên nội
    dung tin nhắn (gộp cả kết quả phân tích ảnh nếu có) để làm ngữ cảnh cho
    standard_llm_node. Nếu Qdrant lỗi/chưa cấu hình, trả về context rỗng -
    không làm fail request chính (fail-soft).
    """
    raw_message = state.get("raw_message", "")
    vision_summary = state.get("vision_summary", "")
    project_id = state.get("project_id", "")
    combined_query = f"{raw_message} {vision_summary}".strip()

    if not combined_query:
        return {
            "rag_context": "",
            "rag_sources": [],
            "routing_log": ["[rag_retrieval_node] Không có nội dung để truy hồi."],
        }

    try:
        from src.rag_retriever import retrieve_dual_rag_context

        result = retrieve_dual_rag_context(combined_query, project_id=project_id)
        log_line = f"[rag_retrieval_node] project_id={project_id} tìm thấy {len(result['sources'])} nguồn tham khảo"
        logger.info(log_line)
        return {
            "rag_context": result["context_text"],
            "rag_sources": result["sources"],
            "routing_log": [log_line],
        }
    except Exception as exc:  # noqa: BLE001 - RAG là tầng bổ trợ, không được làm fail request
        logger.warning("[rag_retrieval_node] Lỗi Dual-RAG, tiếp tục không có ngữ cảnh: %s", exc)
        return {
            "rag_context": "",
            "rag_sources": [],
            "routing_log": [f"[rag_retrieval_node] LỖI (bỏ qua): {exc}"],
        }


def emergency_router_node(state: AgentState) -> AgentState:
    """
    Rule Layer - Emergency Router: quét raw_message + vision_summary theo danh
    sách từ khóa khẩn cấp (tụt áp / rò rỉ / nổ / cháy / quá áp / mất nước / tắt
    lửa buồng đốt...). Nhờ vậy, nếu Vision Model mô tả ảnh có dấu hiệu sự cố,
    hệ thống cũng tự động kích hoạt khẩn cấp kể cả khi người dùng không gõ từ
    khóa đó. Áp dụng cho MỌI role, kể cả ADMIN - an toàn không có ngoại lệ.
    """
    combined_text = _normalize(state.get("raw_message", "") + " " + state.get("vision_summary", ""))
    found = [kw for kw in EMERGENCY_KEYWORDS if _normalize(kw) in combined_text]
    is_emergency = len(found) > 0

    log_line = f"[emergency_router_node] is_emergency={is_emergency} keywords_found={found}"
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
    """
    keywords = state.get("keywords_found", [])
    msg = (
        f"⚠️ CẢNH BÁO KHẨN CẤP: Phát hiện từ khóa sự cố [{', '.join(keywords)}]. "
        "Hệ thống đã ghi nhận ưu tiên cao nhất. Kỹ sư vận hành vui lòng kiểm tra "
        "hiện trường ngay lập tức và thực hiện quy trình ứng phó khẩn cấp."
    )
    logger.critical("[emergency_handler_node] %s", msg)
    return {
        "final_response": msg,
        "routing_log": [f"[emergency_handler_node] escalated keywords={keywords}"],
    }


def standard_llm_node(state: AgentState) -> AgentState:
    """
    Xử lý qua Groq LLM (Llama-3.3-70B) cho MỌI role (OPERATOR lẫn ADMIN), có
    đưa vào ngữ cảnh Dual-RAG (nếu tìm thấy) và kết quả phân tích ảnh (nếu
    có). ADMIN nhận system prompt điều chỉnh (Admin God Mode): thông tin chi
    tiết hơn, không rào trước đón sau kiểu "nên hỏi kỹ sư khác".

    Mọi lời gọi mạng đều bọc trong try/except + retry/sleep theo chuẩn công
    nghiệp, không được phép làm sập hệ thống SCADA nếu mất mạng tạm thời.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_message = state.get("raw_message", "")
    vision_summary = state.get("vision_summary", "")
    rag_context = state.get("rag_context", "")
    is_admin = state.get("user_role") == "ADMIN"

    # Loại thiết bị cụ thể của dự án (khai báo qua lệnh /set_boiler_type trên Telegram) -
    # giúp AI nói cụ thể "lò ghi bậc thang của dự án", thay vì nói chung chung "hệ thống lò
    # hơi/lò dầu tải nhiệt". Fail-soft: nếu Supabase lỗi hoặc chưa khai báo, bỏ qua, không
    # chặn luồng trả lời chính.
    boiler_type = ""
    try:
        from src.project_registry import get_boiler_type_for_group

        boiler_type = get_boiler_type_for_group(state.get("group_id", ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[standard_llm_node] Không lấy được boiler_type (bỏ qua): %s", exc)

    base_prompt = """BẠN LÀ AI:
Bạn là kỹ sư tư vấn kỹ thuật cấp cao, chuyên sâu về lò hơi (steam boiler) và lò dầu tải
nhiệt (thermal oil heater) công nghiệp, đang hỗ trợ đội vận hành nhà máy qua kênh chat
nội bộ. Bạn có nhiều năm kinh nghiệm thực chiến, không chỉ trả lời lý thuyết sách vở.

QUY TRÌNH BẮT BUỘC TRƯỚC KHI TRẢ LỜI (thực hiện trong đầu, không cần in ra từng bước):
1. Đọc kỹ toàn bộ phần "TÀI LIỆU THAM KHẢO NỘI BỘ" (nếu có) được cung cấp bên dưới câu
   hỏi - đây là dữ liệu nội bộ ưu tiên cao nhất, phản ánh đúng thực tế của nhà máy.
2. Xác định câu hỏi có trả lời được bằng tài liệu tham khảo không:
   - Nếu CÓ đoạn tài liệu liên quan: dùng ĐÚNG số liệu/ngưỡng/quy trình như trong tài
     liệu (không làm tròn, không diễn giải lại số liệu an toàn), và trích dẫn theo số
     thứ tự tài liệu (ví dụ: "theo Tài liệu 2...").
   - Nếu tài liệu chỉ liên quan MỘT PHẦN: dùng phần liên quan, và nói rõ phần bạn đang bổ
     sung là suy luận từ kiến thức chuyên môn chung, không phải từ tài liệu nội bộ.
   - Nếu KHÔNG có tài liệu nào liên quan: trả lời dựa trên kiến thức kỹ thuật lò hơi công
     nghiệp phổ quát, nhưng PHẢI nói rõ ngay: "Không tìm thấy tài liệu nội bộ liên quan,
     đây là kiến thức chuyên môn chung, cần đối chiếu lại với SOP thực tế của nhà máy."
3. Nếu các đoạn tài liệu có vẻ mâu thuẫn nhau, hãy chỉ ra sự mâu thuẫn đó thay vì tự ý
   chọn 1 đoạn và bỏ qua đoạn còn lại.
4. CẢNH GIÁC với các sự cố có TRIỆU CHỨNG GIỐNG NHAU nhưng NGUYÊN NHÂN/BẢN CHẤT khác nhau
   (ví dụ: "đóng keo xỉ" (clinkering/slagging) và "sụt tường buồng đốt" đều có thể gây ra
   dấu hiệu tương tự như tiếng động lớn, rung chấn, cản trở luồng khí - nhưng là 2 sự cố
   HOÀN TOÀN KHÁC NHAU về nguyên nhân và cách xử lý). Đọc kỹ để xác định ĐÚNG tên sự cố mà
   tài liệu mô tả trước khi kết luận, KHÔNG suy diễn tên sự cố chỉ từ triệu chứng bề ngoài
   nếu tài liệu đã nêu rõ tên chính xác.
5. TUYỆT ĐỐI không tự bịa số liệu (áp suất, nhiệt độ, mã quy trình...) không có trong tài
   liệu hoặc không thuộc kiến thức chuyên môn đã được kiểm chứng.

PHONG CÁCH TRẢ LỜI (tránh trả lời máy móc, cứng nhắc):
- Trả lời như một kỹ sư đang tư vấn trực tiếp cho đồng nghiệp: tự nhiên, đi thẳng vào
  trọng tâm, KHÔNG lặp lại cấu trúc rập khuôn "Câu hỏi của bạn là..." mỗi lần.
- Ưu tiên viết thành câu văn liền mạch; chỉ dùng gạch đầu dòng khi thực sự liệt kê từ 3
  bước/mục trở lên.
- Độ dài tương xứng độ phức tạp: câu hỏi đơn giản trả lời ngắn gọn; câu hỏi kỹ thuật phức
  tạp có thể trả lời dài hơn, có cấu trúc rõ ràng.
- LUÔN trả lời bằng tiếng Việt có dấu đầy đủ.
- TUYỆT ĐỐI KHÔNG dùng ký hiệu định dạng Markdown (###, **, *, _, dấu backtick, gạch đầu
  dòng "-" hay bảng dạng "| ô | ô |"). Kênh Telegram hiển thị các ký hiệu này y nguyên
  thành chữ thô, làm câu trả lời trông rối và vô nghĩa. Khi cần liệt kê, dùng số thứ tự
  viết liền trong câu (ví dụ "Thứ nhất, ... Thứ hai, ...") hoặc xuống dòng đơn giản kèm số
  "1)", "2)"; khi cần nhấn mạnh thì dùng từ ngữ để nhấn mạnh, không dùng **in đậm**.

RÀNG BUỘC AN TOÀN:
- Không suy đoán các thông số an toàn quan trọng (áp suất tối đa, ngưỡng cảnh báo, quy
  trình khẩn cấp) nếu không có căn cứ rõ ràng.
- Nếu câu hỏi liên quan an toàn mà thông tin không đủ rõ, ưu tiên khuyến nghị thận trọng
  (dừng vận hành, kiểm tra trực tiếp hiện trường) hơn là đưa ra câu trả lời có thể sai."""

    if boiler_type:
        base_prompt += (
            f"\n\nTHIẾT BỊ CỦA DỰ ÁN NÀY: {boiler_type}\n"
            "Khi trả lời, hãy nói CỤ THỂ theo đúng loại thiết bị này (ví dụ nếu là 'lò ghi "
            "bậc thang' thì nói 'lò ghi bậc thang', không nói chung chung 'hệ thống lò hơi/lò "
            "dầu tải nhiệt'). Nếu tài liệu tham khảo hoặc kiến thức chung không khớp với đúng "
            "loại thiết bị này, hãy nói rõ giới hạn đó."
        )

    if is_admin:
        system_prompt = (
            base_prompt
            + "\n\nNGƯỜI HỎI: Kỹ sư trưởng Lê Đức Long (quyền ADMIN). Được phép đi sâu vào chi "
            "tiết kỹ thuật, số liệu thô, phương án xử lý cụ thể. KHÔNG cần đề xuất 'nên liên hệ "
            "kỹ sư khác kiểm tra' vì chính người hỏi là kỹ sư trưởng. Nếu tài liệu tham khảo "
            "không đủ để trả lời chắc chắn, hãy nói rõ đang thiếu thông tin gì."
        )
    else:
        system_prompt = (
            base_prompt
            + "\n\nNGƯỜI HỎI: kỹ sư/nhân viên vận hành (OPERATOR). Nếu không chắc chắn, hãy nói "
            "rõ và đề xuất liên hệ kỹ sư trưởng hoặc kiểm tra trực tiếp hiện trường."
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
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
            )
            content = getattr(response, "content", str(response))
            # Không gắn tiền tố "[ADMIN MODE]" vào nội dung hiển thị cho người dùng - lặp lại
            # mỗi tin nhắn tạo cảm giác máy móc/cứng nhắc. Trạng thái admin vẫn được ghi đầy
            # đủ vào routing_log/Supabase để audit, chỉ không hiện trong câu trả lời chat.
            logger.info("[standard_llm_node] thành công ở lần thử %s (admin=%s)", attempt, is_admin)
            return {
                "final_response": content,
                "routing_log": [f"[standard_llm_node] OK attempt={attempt} admin={is_admin}"],
            }
        except Exception as exc:  # noqa: BLE001 - phải bắt mọi lỗi, không được crash
            last_error = exc
            logger.warning(
                "[standard_llm_node] Lỗi mạng/API lần %s/%s: %s. Ngủ đông %ss rồi thử lại.",
                attempt,
                MAX_RETRY,
                exc,
                RETRY_SLEEP_SECONDS,
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
    graph.add_node("rag_retrieval_node", rag_retrieval_node)
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
    graph.add_edge("vision_analysis_node", "rag_retrieval_node")
    graph.add_edge("rag_retrieval_node", "emergency_router_node")

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
