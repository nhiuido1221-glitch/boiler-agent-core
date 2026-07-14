"""
telegram_bot.py
-----------------
Vệ tinh giao tiếp Telegram cho Boiler Agent. Chạy long-polling bằng
pyTelegramBotAPI (telebot), gọi trực tiếp LangGraph đã compile (không qua
HTTP) để giảm độ trễ, rồi trả kết quả về đúng group/chat Telegram.

Lệnh hỗ trợ:
  /hoi <câu hỏi>                          - GỌI BOT trong group chat (xem bên dưới)
  /new_project <project_id> <Tên dự án>   - (ADMIN) gán group hiện tại vào 1 dự án
  /my_project                             - xem dự án đang gán cho group hiện tại
  /help                                   - xem hướng dẫn nhanh

Chống spam trong group (Cải tiến #4): bot KHÔNG trả lời mọi tin nhắn trong
group. Chỉ xử lý khi:
  (a) tin nhắn bắt đầu bằng "/hoi", hoặc
  (b) tin nhắn là REPLY trực tiếp vào 1 tin nhắn trước đó của chính bot
      (cho phép hội thoại tiếp nối tự nhiên mà không cần gõ lại /hoi mỗi lần).
Trong chat riêng (DM 1-1 với bot), luôn xử lý bình thường, không cần lệnh.

Upload tài liệu (Cải tiến #3, chỉ ADMIN): gửi file .txt / .pdf / .docx, KHÔNG
cần ghi caption (một số client Telegram không cho thêm caption khi gửi file).
Bot sẽ hỏi lại bằng 2 nút bấm (inline keyboard): "Kho DÙNG CHUNG" hay
"Dự án hiện tại" - ADMIN chỉ cần bấm 1 nút, không cần gõ gì thêm.

Thông báo tiến trình (Cải tiến #2): mọi bước xử lý file đều có thông báo rõ
ràng (nhận file -> chọn kho -> đang nạp -> kết quả thành công/thất bại), để
không bao giờ có cảm giác "im lặng không phản hồi".

Bảo mật: chỉ tin tưởng quyền ADMIN nếu telegram user_id khớp ADMIN_ID trong
.env. Mọi user khác mặc định là OPERATOR.

Nguyên tắc công nghiệp: bot polling phải tự phục hồi khi mất mạng - toàn bộ
vòng lặp được bọc try/except + sleep/retry, không bao giờ để thread chết hẳn.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger("telegram_bot")

ADMIN_ID = os.getenv("ADMIN_ID", "")

# Lệnh "dạy" AI qua Telegram (chỉ ADMIN) - chấp nhận vài biến thể gõ (có/không dấu)
# để không đòi hỏi anh Long phải gõ chính xác 100% có dấu mỗi lần.
ADMIN_TEACH_PREFIXES = ("lưu lại:", "luu lai:", "ghi nhớ:", "ghi nho:")
RETRY_SLEEP_SECONDS = 5
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20MB, khớp giới hạn Telegram Bot API cho getFile
GROUP_COMMAND_PREFIX = "/hoi"  # từ khóa gọi bot trong group chat, tránh spam
TELEGRAM_MAX_MESSAGE_CHARS = 3800  # giới hạn thật của Telegram là 4096, chừa margin an toàn

_bot_instance = None
_bot_thread: Optional[threading.Thread] = None

# Lưu tạm thông tin file đang chờ ADMIN chọn kho lưu (kho chung / dự án riêng).
# Key = upload_id ngắn gọn (đưa vào callback_data, giới hạn 64 byte của Telegram).
_pending_uploads: dict[str, dict] = {}
_pending_uploads_lock = threading.Lock()

# Lưu tạm dự án đang chờ ADMIN XÁC NHẬN xoá (qua nút bấm) - /delete_project là thao
# tác PHÁ HUỶ DỮ LIỆU VĨNH VIỄN nên bắt buộc phải qua bước xác nhận riêng, không xoá
# ngay khi gõ lệnh. Mất khi bot khởi động lại (an toàn hơn: 1 phiên xác nhận "treo"
# quá lâu qua nhiều lần restart không nên còn hiệu lực).
_pending_deletes: dict[str, dict] = {}
_pending_deletes_lock = threading.Lock()


def _resolve_role(telegram_user_id: str) -> str:
    """ADMIN chỉ được cấp khi telegram_user_id khớp chính xác ADMIN_ID cấu hình."""
    if ADMIN_ID and telegram_user_id == ADMIN_ID:
        return "ADMIN"
    return "OPERATOR"


def _split_long_message(text: str, max_chars: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """
    Chia 1 chuỗi trả lời dài thành nhiều phần <= max_chars để không vượt giới hạn
    4096 ký tự của Telegram Bot API (lỗi "Bad Request: message is too long").
    Ưu tiên cắt tại ranh giới đoạn văn ("\n\n"), rồi tới dòng ("\n"), rồi tới
    khoảng trắng - chỉ cắt cứng giữa từ khi không còn lựa chọn nào khác, để giữ
    câu văn liền mạch nhất có thể giữa các phần.
    """
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind("\n\n", 0, max_chars)
        if cut == -1:
            cut = remaining.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut == -1 or cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)

    total = len(parts)
    return [f"[{idx}/{total}]\n{part}" if total > 1 else part for idx, part in enumerate(parts, start=1)]


def _send_long_reply(bot, message, text: str, max_retry: int = 3) -> None:
    """
    Gửi trả lời có thể dài hơn giới hạn Telegram - tự tách thành nhiều tin
    nhắn liên tiếp.

    Nguyên nhân sự cố "bot im lặng không trả lời" (đã xác nhận qua log Render
    thật): pyTelegramBotAPI KHÔNG có timeout mặc định cho các lệnh gửi tin
    (reply_to/send_message) - nếu mạng tới api.telegram.org chập chờn tạm
    thời, lệnh gửi treo VÔ THỜI HẠN, không raise lỗi, nên try/except bên
    ngoài không bao giờ được kích hoạt để báo lỗi hay thử lại. Đã vá 2 lớp:
      1) Timeout toàn cục cho MỌI lệnh gọi Telegram API (xem apihelper.
         CONNECT_TIMEOUT/READ_TIMEOUT được set 1 lần lúc khởi tạo bot trong
         _build_bot()) - biến "treo vĩnh viễn" thành "báo lỗi có giới hạn
         thời gian", đúng chuẩn công nghiệp bắt buộc của dự án.
      2) Retry + sleep ngay tại đây (giống mẫu đã dùng cho Groq/Gemini) - 1
         lần chập chờn thoáng qua sẽ tự phục hồi, không cần Kỹ sư Long phải
         gửi lại tin nhắn.
    """
    chunks = _split_long_message(text)

    def _send_one(chunk: str, is_first: bool) -> None:
        last_error: Exception | None = None
        for attempt in range(1, max_retry + 1):
            try:
                if is_first:
                    bot.reply_to(message, chunk)
                else:
                    bot.send_message(message.chat.id, chunk)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "[_send_long_reply] Lỗi gửi Telegram lần %s/%s: %s. Thử lại sau %ss.",
                    attempt, max_retry, exc, RETRY_SLEEP_SECONDS,
                )
                if attempt < max_retry:
                    time.sleep(RETRY_SLEEP_SECONDS)
        logger.error(
            "[_send_long_reply] Gửi thất bại sau %s lần thử (chat_id=%s): %s",
            max_retry, getattr(message.chat, "id", "?"), last_error,
        )

    _send_one(chunks[0], is_first=True)
    for chunk in chunks[1:]:
        _send_one(chunk, is_first=False)




def _extract_image_urls(bot, message) -> list[str]:
    """
    Nếu tin nhắn có ảnh đính kèm, lấy URL file trực tiếp từ Telegram (public
    file link theo bot token) để đưa vào Vision Model. Nếu lỗi lấy file
    (mạng chập chờn), trả về danh sách rỗng - không chặn luồng xử lý text.
    """
    if not message.photo:
        return []
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
        return [file_url]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không lấy được URL ảnh từ Telegram: %s", exc)
        return []


def _parse_group_command(text: str, bot_username: str) -> tuple[bool, str]:
    """
    Kiểm tra tin nhắn có bắt đầu bằng lệnh gọi bot (/hoi, hoặc /hoi@ten_bot khi
    group có nhiều bot) không. Trả về (is_command, phần_text_sau_lệnh).
    """
    parts = (text or "").strip().split(maxsplit=1)
    if not parts:
        return False, ""
    first_word = parts[0].lower()
    is_command = first_word == GROUP_COMMAND_PREFIX or first_word == f"{GROUP_COMMAND_PREFIX}@{bot_username.lower()}"
    remaining = parts[1] if len(parts) > 1 else ""
    return is_command, remaining


def _build_bot(compiled_graph):
    import telebot
    from telebot import types, apihelper

    # FIX SU CO "BOT IM LANG KHONG TRA LOI" (xac nhan qua log Render that:
    # request toi Supabase thanh cong, nhung sau do khong co gi - khong loi,
    # khong phan hoi, treo vo thoi han). Nguyen nhan: pyTelegramBotAPI khong
    # co timeout mac dinh cho cac lenh GOI RA (reply_to/send_message/get_file/
    # get_me...) - 1 lan mang chap chon toi api.telegram.org la treo thread xu
    # ly tin nhan do mai mai, try/except ben ngoai khong bao gio duoc kich
    # hoat vi khong co exception nao duoc raise ra de bat. Set timeout toan
    # cuc o day ap dung cho MOI lenh goi Telegram API trong suot vong doi bot,
    # bien "treo vinh vien" thanh "bao loi co gioi han thoi gian" - dung
    # nguyen tac cong nghiep bat buoc cua du an (khong duoc im lang khi mat
    # mang tam thoi).
    # Da tang tu (15, 20) -> (30, 90): log Render thuc te cho thay send_photo() (gui
    # anh SCADA) lien tuc bao 'Connection aborted... write operation timed out' khi
    # mang tam thoi cham/bi canh tranh boi request /invoke khac chay song song -
    # 20s la qua ngan de day het 1 anh PNG chup man hinh len Telegram trong dieu
    # kien do. READ_TIMEOUT can lon hon CONNECT_TIMEOUT nhieu vi day la thoi gian
    # cho ca qua trinh upload + Telegram xu ly xong, khong chi cho luc ket noi.
    apihelper.CONNECT_TIMEOUT = 30
    apihelper.READ_TIMEOUT = 90

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token or token.startswith("xxxxxx"):
        raise RuntimeError("TELEGRAM_BOT_TOKEN chưa được cấu hình đúng trong .env")

    bot = telebot.TeleBot(token, parse_mode=None)

    # Lấy thông tin bot 1 lần lúc khởi động (id + username) để nhận diện reply/mention,
    # tránh gọi API get_me() lặp lại mỗi tin nhắn.
    bot_me = bot.get_me()
    bot_user_id = bot_me.id
    bot_username = bot_me.username or ""
    logger.info("Telegram bot đã xác thực: @%s (id=%s)", bot_username, bot_user_id)

    # --------------------------------------------------------------------
    # Lệnh trợ giúp
    # --------------------------------------------------------------------
    @bot.message_handler(commands=["help", "start"])
    def handle_help(message):
        bot.reply_to(
            message,
            "🤖 Boiler Agent - hướng dẫn nhanh:\n\n"
            f"• Trong group: gõ '{GROUP_COMMAND_PREFIX} <câu hỏi>' để hỏi bot (hoặc reply "
            "thẳng vào tin nhắn trước đó của bot để hỏi tiếp, không cần gõ lại lệnh).\n"
            "• Trong chat riêng với bot: nhắn bình thường, không cần lệnh.\n"
            "• /new_project <mã> <Tên dự án> - (ADMIN) gán group này vào 1 dự án riêng.\n"
            "• /set_boiler_type <mô tả loại lò> - (ADMIN) khai báo loại thiết bị cụ thể của "
            "dự án (vd: 'Lò hơi ống lửa 10 tấn/h đốt trấu'), để AI trả lời đúng loại thiết bị "
            "thay vì nói chung chung.\n"
            "• /my_project - xem group này đang thuộc dự án nào, loại lò gì.\n"
            "• /list_projects - (ADMIN) xem TOÀN BỘ dự án đã khởi tạo, gọi được từ chat riêng, "
            "không cần vào từng nhóm.\n"
            "• /list_docs [mã dự án] - xem danh sách tài liệu; không gõ mã thì mặc định là dự án "
            "của group đang chat, gõ thêm mã (vd '/list_docs nhamay_binhduong') để xem dự án khác.\n"
            "• Gửi file .txt/.pdf/.docx - (ADMIN) nạp tài liệu; bot cho chọn NƠI LƯU bằng nút bấm, "
            "kể cả từ chat riêng cũng chọn được thẳng dự án mong muốn, không cần vào đúng nhóm đó.\n"
            "• LƯU LẠI: <nội dung> - (ADMIN) dạy AI 1 kinh nghiệm thực tế, lưu ngay vào kho.\n"
            "• /delete_project <mã> - (ADMIN) XOÁ VĨNH VIỄN 1 dự án (cả tài liệu), có xác nhận qua nút bấm.\n"
            "• /scada - (ADMIN) yêu cầu máy trạm chụp ảnh SCADA + phân tích dữ liệu từ đầu ca NGAY LẬP TỨC "
            "(không cần đợi chu kỳ tự động), kết quả gửi thẳng vào group này trong ít phút.",
        )

    # --------------------------------------------------------------------
    # Lệnh quản lý dự án (chỉ ADMIN)
    # --------------------------------------------------------------------
    @bot.message_handler(commands=["new_project"])
    def handle_new_project(message):
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới được tạo/gán dự án.")
                return

            parts = (message.text or "").split(maxsplit=2)
            if len(parts) < 3:
                bot.reply_to(
                    message,
                    "Cú pháp: /new_project <ma_du_an> <Tên dự án>\n"
                    "Ví dụ: /new_project nhamay_binhduong Nhà máy Bình Dương",
                )
                return

            project_id, project_name = parts[1].strip(), parts[2].strip()
            group_id = str(message.chat.id)

            from src.project_registry import register_project

            ok = register_project(
                project_id=project_id,
                project_name=project_name,
                group_id=group_id,
                created_by=telegram_user_id,
            )
            if ok:
                bot.reply_to(
                    message,
                    f"✅ Đã gán group này vào dự án '{project_name}' (mã: {project_id}).\n"
                    "Từ giờ tài liệu upload trong group này sẽ thuộc riêng dự án này, "
                    "không lẫn với dự án khác.",
                )
            else:
                bot.reply_to(message, "❌ Lỗi khi ghi vào Supabase, kiểm tra lại cấu hình SUPABASE_URL/KEY.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /new_project: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    @bot.message_handler(commands=["set_boiler_type"])
    def handle_set_boiler_type(message):
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới được khai báo loại thiết bị.")
                return

            parts = (message.text or "").split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                bot.reply_to(
                    message,
                    "Cú pháp: /set_boiler_type <mô tả loại thiết bị>\n"
                    "Ví dụ: /set_boiler_type Lò hơi ống lửa 10 tấn/h đốt trấu\n"
                    "Ví dụ: /set_boiler_type Lò dầu tải nhiệt Q=3.000.000 Kcal/h, đốt than",
                )
                return

            boiler_type = parts[1].strip()
            group_id = str(message.chat.id)

            from src.project_registry import set_boiler_type

            ok = set_boiler_type(group_id=group_id, boiler_type=boiler_type)
            if ok:
                bot.reply_to(
                    message,
                    f"✅ Đã cập nhật loại thiết bị cho dự án này: '{boiler_type}'.\n"
                    "Từ giờ AI sẽ trả lời cụ thể theo đúng loại thiết bị này.",
                )
            else:
                bot.reply_to(
                    message,
                    "❌ Chưa gán được. Group này có thể chưa chạy /new_project, hoặc lỗi kết nối "
                    "Supabase - kiểm tra lại.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /set_boiler_type: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    @bot.message_handler(commands=["my_project"])
    def handle_my_project(message):
        try:
            group_id = str(message.chat.id)
            from src.project_registry import get_project_info

            info = get_project_info(group_id)
            if info:
                boiler_type = info.get("boiler_type") or "(chưa khai báo - dùng /set_boiler_type để thêm)"
                bot.reply_to(
                    message,
                    f"Group này thuộc dự án '{info.get('project_name')}' (mã: {info.get('project_id')}).\n"
                    f"Loại thiết bị: {boiler_type}",
                )
            else:
                bot.reply_to(
                    message,
                    "Group này chưa được gán dự án nào, đang dùng kho tài liệu mặc định. "
                    "Dùng /new_project <ma_du_an> <Tên dự án> để tạo dự án riêng.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /my_project: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    @bot.message_handler(commands=["list_docs"])
    def handle_list_docs(message):
        try:
            # Cho phép xem tài liệu của BẤT KỲ dự án nào bằng cách gõ thêm mã dự án
            # (vd "/list_docs nhamay_binhduong") - không cần đứng trong đúng nhóm đó.
            # Không truyền gì thì vẫn mặc định là dự án của group đang chat như cũ.
            parts = (message.text or "").split(maxsplit=1)
            explicit_project_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

            if explicit_project_id:
                project_id = explicit_project_id
            else:
                group_id = str(message.chat.id)
                from src.project_registry import get_project_id_for_group

                project_id = get_project_id_for_group(group_id)

            from src.rag_retriever import list_documents

            docs = list_documents(project_id)

            lines = [f"📚 Tài liệu cho dự án '{project_id}' (gồm cả kho dùng chung):"]
            total = 0
            section_titles = {"knowledge": "Kiến thức / SOP", "history": "Lịch sử sự cố"}
            for label, title in section_titles.items():
                sources = docs.get(label, {})
                if not sources:
                    continue
                lines.append(f"\n{title}:")
                for src, count in sorted(sources.items()):
                    lines.append(f"  • {src} ({count} đoạn)")
                    total += count

            if total == 0:
                lines.append("\n(Chưa có tài liệu nào. Gửi file .txt/.pdf/.docx để nạp - chỉ ADMIN.)")

            _send_long_reply(bot, message, "\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /list_docs: %s", exc)
            bot.reply_to(message, f"❌ Lỗi khi lấy danh sách tài liệu: {exc}")

    @bot.message_handler(commands=["list_projects"])
    def handle_list_projects(message):
        """
        Liệt kê TOÀN BỘ dự án đã khởi tạo (mọi nhóm), gọi được từ BẤT KỲ ĐÂU (kể cả
        chat riêng) - không cần đứng trong đúng nhóm Telegram của dự án đó. Chỉ ADMIN
        xem được: danh sách này lộ ra tên/mã của MỌI nhóm/nhà máy đang dùng hệ thống,
        thành viên nhóm A không nên biết nhóm B/C/D nào đang tồn tại.
        """
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới xem được danh sách toàn bộ dự án.")
                return

            from src.project_registry import list_all_projects

            projects = list_all_projects()
            if not projects:
                bot.reply_to(
                    message,
                    "Chưa có dự án nào được khởi tạo. Dùng /new_project <mã> <Tên dự án> "
                    "trong 1 nhóm Telegram để tạo dự án đầu tiên.",
                )
                return

            lines = [f"📋 Danh sách {len(projects)} dự án đã khởi tạo:"]
            for p in projects:
                boiler_type = p.get("boiler_type") or "(chưa khai báo)"
                lines.append(
                    f"\n• {p.get('project_name') or '(chưa đặt tên)'}"
                    f"\n  Mã: {p.get('project_id')} | Nhóm: {p.get('group_id')} | Loại lò: {boiler_type}"
                )
            lines.append("\n📌 Ngoài ra luôn có 'Kho DÙNG CHUNG' (mã: shared) - tài liệu áp dụng cho mọi dự án.")

            _send_long_reply(bot, message, "".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /list_projects: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    # --------------------------------------------------------------------
    # /scada (chỉ ADMIN) - yêu cầu máy trạm chụp ảnh SCADA + phân tích tức thời
    # --------------------------------------------------------------------
    @bot.message_handler(commands=["scada"])
    def handle_scada_instant_report(message):
        """
        ADMIN gõ /scada trong group -> Core tra bảng 'stations' tìm máy trạm gắn
        với group_id này -> chèn 1 lệnh 'instant_report' vào station_commands
        (Supabase). Máy trạm đang chạy sẽ tự poll thấy lệnh này trong tối đa
        ~30s (chu kỳ poll_interval_seconds cấu hình ở máy trạm), chụp ảnh SCADA
        NGAY tại thời điểm đó + tổng hợp dữ liệu từ đầu ca hiện tại đến lúc yêu
        cầu, rồi tự gọi /invoke với notify_telegram=True để đẩy kết quả (kèm
        ảnh) vào ĐÚNG group này - không cần ADMIN tự vào Supabase Studio chạy
        SQL tay như lúc kiểm thử ban đầu.

        Đây là hành động "yêu cầu", KHÔNG phải kết quả tức thì trong cùng 1 lượt
        trả lời - bot chỉ xác nhận đã gửi yêu cầu thành công, kết quả THẬT sẽ đến
        sau (từ /invoke của máy trạm) như một tin nhắn RIÊNG.
        """
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới được yêu cầu chụp ảnh SCADA tức thời.")
                return

            group_id = str(message.chat.id)
            from src.supabase_logger import queue_station_command

            ok, info = queue_station_command(group_id, "instant_report")
            if ok:
                bot.reply_to(
                    message,
                    f"📸 Đã gửi yêu cầu tới máy trạm ({info}). Máy trạm sẽ chụp ảnh SCADA + "
                    "tổng hợp dữ liệu từ đầu ca đến giờ, thường mất tối đa khoảng 1 phút "
                    "(chu kỳ máy trạm kiểm tra lệnh mới). Kết quả sẽ tự động gửi vào đây "
                    "ngay khi xong, không cần gõ lệnh lại.",
                )
            else:
                bot.reply_to(
                    message,
                    f"❌ Không gửi được yêu cầu tới máy trạm: {info}\n"
                    "Kiểm tra máy trạm đã đăng ký đúng group này trong bảng 'stations' "
                    "(cột group_id) và đang ở trạng thái is_active=true chưa.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /scada: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    # --------------------------------------------------------------------
    # Nút "Đã xác nhận" đính kèm tin cảnh báo khẩn cấp (is_emergency=True) do
    # send_notification() tự động gắn - xem send_notification() bên dưới.
    # ADMIN bấm nút này thay vì phải tự gõ SQL/lệnh - chèn thẳng lệnh
    # 'ack_alert' vào station_commands, máy trạm tự nhặt trong tối đa
    # poll_interval_seconds (mặc định 30s) và tạm ngưng gửi cảnh báo lặp lại
    # trong ack_mute_minutes phút (mặc định 30 phút, tự động bật lại sau đó).
    # --------------------------------------------------------------------
    @bot.callback_query_handler(func=lambda call: call.data.startswith("ackalert:"))
    def handle_ack_alert_button(call):
        try:
            _, group_id = call.data.split(":", 1)

            # BAO MAT: kiem tra nguoi BAM NUT ngay bay gio (call.from_user.id) -
            # nut nay hien thi cho CA GROUP thay (Telegram inline keyboard khong
            # rieng tu theo nguoi nhan tin canh bao), giong nguyen tac da ap dung
            # cho nut xoa du an / chon kho luu tai lieu o tren.
            clicker_id = str(call.from_user.id) if call.from_user else ""
            if _resolve_role(clicker_id) != "ADMIN":
                bot.answer_callback_query(call.id, "Chỉ ADMIN mới được xác nhận cảnh báo.")
                return

            from src.supabase_logger import queue_station_command

            ok, info = queue_station_command(group_id, "ack_alert")

            if ok:
                bot.answer_callback_query(call.id, "Đã xác nhận, tạm ngưng cảnh báo lặp lại.")
                confirm_note = (
                    f"\n\n✅ ADMIN đã xác nhận lúc "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Tạm ngưng cảnh báo lặp lại."
                )
                try:
                    if call.message.content_type == "photo":
                        old_caption = call.message.caption or ""
                        bot.edit_message_caption(
                            caption=(old_caption + confirm_note)[:1024],
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                        )
                    else:
                        old_text = call.message.text or ""
                        bot.edit_message_text(
                            (old_text + confirm_note)[:4096],
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                        )
                except Exception as exc:  # noqa: BLE001 - sua tin chi la trang tri, khong duoc lam fail xac nhan
                    logger.warning("[handle_ack_alert_button] Không sửa được tin gốc (bỏ qua, đã ghi nhận ack): %s", exc)
            else:
                bot.answer_callback_query(call.id, f"Lỗi: {info[:180]}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý nút xác nhận cảnh báo: %s", exc)
            try:
                bot.answer_callback_query(call.id, f"Lỗi: {exc}"[:200])
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------------
    # Xoá dự án (chỉ ADMIN) - PHÁ HUỶ DỮ LIỆU VĨNH VIỄN, bắt buộc xác nhận qua nút bấm
    # --------------------------------------------------------------------
    @bot.message_handler(commands=["delete_project"])
    def handle_delete_project(message):
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới được xoá dự án.")
                return

            parts = (message.text or "").split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                bot.reply_to(message, "Cú pháp: /delete_project <ma_du_an>\nVí dụ: /delete_project nhamay_binhduong")
                return

            target_project_id = parts[1].strip()

            from src.project_registry import SHARED_PROJECT_ID

            if target_project_id == SHARED_PROJECT_ID:
                bot.reply_to(
                    message,
                    f"⛔ '{target_project_id}' là kho DÙNG CHUNG cho mọi dự án, không thể xoá qua lệnh này.",
                )
                return

            from src.project_registry import get_project_group_ids
            from src.rag_retriever import count_project_chunks

            group_ids = get_project_group_ids(target_project_id)
            counts = count_project_chunks(target_project_id)
            total_chunks = counts.get("knowledge", 0) + counts.get("history", 0)

            if not group_ids and total_chunks == 0:
                bot.reply_to(message, f"Không tìm thấy dữ liệu nào cho mã dự án '{target_project_id}'.")
                return

            delete_id = uuid.uuid4().hex[:16]
            with _pending_deletes_lock:
                _pending_deletes[delete_id] = {
                    "project_id": target_project_id,
                    "telegram_user_id": telegram_user_id,
                }

            markup = types.InlineKeyboardMarkup()
            markup.add(
                types.InlineKeyboardButton(
                    "🗑️ XÁC NHẬN XOÁ VĨNH VIỄN", callback_data=f"delproj:{delete_id}"
                )
            )
            markup.add(types.InlineKeyboardButton("❌ Huỷ", callback_data=f"delcancel:{delete_id}"))

            bot.reply_to(
                message,
                f"⚠️ SẮP XOÁ VĨNH VIỄN dự án '{target_project_id}':\n"
                f"• {len(group_ids)} nhóm Telegram đang gán vào dự án này sẽ bị gỡ liên kết.\n"
                f"• {counts.get('knowledge', 0)} đoạn tài liệu kỹ thuật + {counts.get('history', 0)} đoạn "
                f"lịch sử sự cố ({total_chunks} tổng) sẽ bị XOÁ KHỎI KHO, KHÔNG THỂ KHÔI PHỤC.\n\n"
                "Bấm nút bên dưới để xác nhận, hoặc bỏ qua tin nhắn này để huỷ.",
                reply_markup=markup,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý /delete_project: %s", exc)
            bot.reply_to(message, f"❌ Đã xảy ra lỗi khi xử lý lệnh: {exc}")

    @bot.callback_query_handler(func=lambda call: call.data.startswith(("delproj:", "delcancel:")))
    def handle_delete_project_choice(call):
        try:
            action, delete_id = call.data.split(":", 1)

            # BẢO MẬT: kiểm tra người BẤM NÚT ngay bây giờ (call.from_user.id), KHÔNG phải
            # người đã gõ lệnh /delete_project lúc trước (info["telegram_user_id"]). Nút bấm
            # hiển thị cho CẢ NHÓM thấy (Telegram inline keyboard không riêng tư theo người
            # gửi) - nếu chỉ kiểm tra người gõ lệnh gốc, bất kỳ thành viên nào trong nhóm
            # cũng có thể bấm thay và vượt qua được lớp ADMIN gate. Áp dụng cho CẢ nút Huỷ,
            # tránh 1 thành viên bất kỳ âm thầm huỷ thao tác của ADMIN.
            clicker_id = str(call.from_user.id) if call.from_user else ""
            if _resolve_role(clicker_id) != "ADMIN":
                bot.answer_callback_query(call.id, "Chỉ ADMIN mới được thao tác.")
                return

            if action == "delcancel":
                with _pending_deletes_lock:
                    _pending_deletes.pop(delete_id, None)
                bot.answer_callback_query(call.id, "Đã huỷ.")
                bot.edit_message_text(
                    chat_id=call.message.chat.id, message_id=call.message.message_id, text="Đã huỷ xoá dự án."
                )
                return

            with _pending_deletes_lock:
                info = _pending_deletes.pop(delete_id, None)

            if not info:
                bot.answer_callback_query(call.id, "Phiên xác nhận đã hết hạn (bot có thể đã khởi động lại). Gõ lại lệnh.")
                return

            target_project_id = info["project_id"]
            bot.answer_callback_query(call.id, "Đang xoá...")
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=f"⏳ Đang xoá vĩnh viễn dự án '{target_project_id}'...",
            )

            from src.project_registry import delete_project_mapping
            from src.rag_retriever import delete_project_chunks

            deleted_chunks = delete_project_chunks(target_project_id)
            deleted_groups = delete_project_mapping(target_project_id)

            total_deleted = deleted_chunks.get("knowledge", 0) + deleted_chunks.get("history", 0)
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=(
                    f"✅ Đã xoá vĩnh viễn dự án '{target_project_id}': "
                    f"{total_deleted} đoạn tài liệu ({deleted_chunks.get('knowledge', 0)} kiến thức + "
                    f"{deleted_chunks.get('history', 0)} lịch sử sự cố), gỡ liên kết {deleted_groups} nhóm."
                ),
            )
            logger.critical(
                "ADMIN %s đã xoá dự án '%s': %s chunk, %s nhóm.",
                info["telegram_user_id"], target_project_id, total_deleted, deleted_groups,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý callback xoá dự án: %s", exc)
            error_text = f"❌ Lỗi khi xoá dự án: {exc}"
            try:
                bot.edit_message_text(
                    chat_id=call.message.chat.id, message_id=call.message.message_id, text=error_text
                )
            except Exception:  # noqa: BLE001
                try:
                    bot.answer_callback_query(call.id, error_text[:200])
                except Exception:  # noqa: BLE001
                    pass

    # --------------------------------------------------------------------
    # Upload tài liệu (chỉ ADMIN) - hỏi lại bằng nút bấm thay vì cần caption
    # --------------------------------------------------------------------
    @bot.message_handler(content_types=["document"])
    def handle_document_upload(message):
        try:
            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            if _resolve_role(telegram_user_id) != "ADMIN":
                bot.reply_to(message, "Chỉ ADMIN mới được nạp tài liệu vào kho kiến thức.")
                return

            filename = message.document.file_name or "unknown_file"
            file_size = message.document.file_size or 0

            if file_size > MAX_UPLOAD_SIZE_BYTES:
                bot.reply_to(message, f"❌ File quá lớn ({file_size / 1024 / 1024:.1f}MB), giới hạn 20MB.")
                return

            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            if ext not in ("txt", "pdf", "docx"):
                bot.reply_to(message, f"❌ Định dạng '.{ext}' chưa hỗ trợ. Chỉ hỗ trợ: .txt, .pdf, .docx")
                return

            from src.project_registry import get_project_id_for_group, list_all_projects

            current_project = get_project_id_for_group(str(message.chat.id))

            # Cho ADMIN chọn THẲNG bất kỳ dự án nào đã khởi tạo, không chỉ dự án của
            # đúng nhóm đang chat - để upload được từ chat riêng mà không cần vào tận
            # nhóm Telegram của dự án đó. Loại bỏ dự án trùng với "dự án hiện tại"
            # (đã có nút riêng) để không hiện 2 nút giống nhau. Giới hạn số nút hiển thị
            # để bàn phím không quá dài (Telegram không giới hạn cứng nhưng UX sẽ tệ).
            all_projects = list_all_projects()
            other_projects_full = [p for p in all_projects if p.get("project_id") != current_project]
            other_projects = other_projects_full[:15]

            upload_id = uuid.uuid4().hex[:16]
            with _pending_uploads_lock:
                _pending_uploads[upload_id] = {
                    "file_id": message.document.file_id,
                    "filename": filename,
                    "group_id": str(message.chat.id),
                    "telegram_user_id": telegram_user_id,
                    "projects": other_projects,
                }

            markup = types.InlineKeyboardMarkup()
            markup.add(
                types.InlineKeyboardButton(
                    "📌 Kho DÙNG CHUNG (mọi dự án)", callback_data=f"kbshared:{upload_id}"
                )
            )
            markup.add(
                types.InlineKeyboardButton(
                    f"📁 Dự án hiện tại ({current_project})", callback_data=f"kbproject:{upload_id}"
                )
            )
            for idx, proj in enumerate(other_projects):
                label = f"📂 {proj.get('project_name') or proj.get('project_id')} ({proj.get('project_id')})"
                markup.add(types.InlineKeyboardButton(label, callback_data=f"kbpick:{upload_id}:{idx}"))

            extra_note = ""
            if len(other_projects_full) > len(other_projects):
                extra_note = "\n(Còn nhiều dự án khác chưa hiện hết - dùng /list_projects để xem đủ danh sách.)"

            bot.reply_to(
                message,
                f"📄 Đã nhận file '{filename}'. Chọn nơi lưu tài liệu:{extra_note}",
                reply_markup=markup,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý nhận file upload: %s", exc)
            try:
                bot.reply_to(message, f"❌ Lỗi khi nhận file: {exc}")
            except Exception:  # noqa: BLE001
                pass

    @bot.callback_query_handler(func=lambda call: call.data.startswith(("kbshared:", "kbproject:", "kbpick:")))
    def handle_upload_choice(call):
        try:
            parts = call.data.split(":")
            scope = parts[0]
            upload_id = parts[1]
            pick_index = int(parts[2]) if scope == "kbpick" and len(parts) > 2 else None

            with _pending_uploads_lock:
                info = _pending_uploads.pop(upload_id, None)

            if not info:
                bot.answer_callback_query(call.id, "Phiên upload đã hết hạn (bot có thể đã khởi động lại). Gửi lại file.")
                return

            # BẢO MẬT: kiểm tra người BẤM NÚT ngay bây giờ (call.from_user.id), KHÔNG phải
            # người đã upload file lúc trước (info["telegram_user_id"]) - xem giải thích đầy
            # đủ ở handle_delete_project_choice. Nút chọn "Kho DÙNG CHUNG"/"Dự án hiện tại"
            # hiển thị cho cả nhóm thấy, không riêng tư theo người upload.
            clicker_id = str(call.from_user.id) if call.from_user else ""
            if _resolve_role(clicker_id) != "ADMIN":
                bot.answer_callback_query(call.id, "Chỉ ADMIN mới được thao tác.")
                return

            bot.answer_callback_query(call.id, "Đang xử lý...")
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=(
                    f"⏳ Đang nạp '{info['filename']}'... (thường mất 10-30 giây; tài liệu lớn/nhiều "
                    "bảng biểu có thể mất tới vài phút nếu gặp giới hạn tốc độ tạm thời từ Gemini - "
                    "hệ thống sẽ tự thử lại, không cần gửi lại file, cứ đợi tin nhắn kết quả)"
                ),
            )

            from src.document_ingest import ingest_document
            from src.project_registry import SHARED_PROJECT_ID, get_project_id_for_group

            if scope == "kbshared":
                target_project_id = SHARED_PROJECT_ID
                scope_label = "kho DÙNG CHUNG"
            elif scope == "kbpick":
                projects = info.get("projects") or []
                if pick_index is None or not (0 <= pick_index < len(projects)):
                    bot.edit_message_text(
                        chat_id=call.message.chat.id,
                        message_id=call.message.message_id,
                        text="❌ Lựa chọn dự án không còn hợp lệ (danh sách có thể đã thay đổi). Gửi lại file.",
                    )
                    return
                picked = projects[pick_index]
                target_project_id = picked.get("project_id")
                scope_label = f"dự án '{picked.get('project_name') or target_project_id}' ({target_project_id})"
            else:
                target_project_id = get_project_id_for_group(info["group_id"])
                scope_label = f"dự án '{target_project_id}'"

            file_info = bot.get_file(info["file_id"])
            file_bytes = bot.download_file(file_info.file_path)

            num_chunks = ingest_document(
                file_bytes=file_bytes, filename=info["filename"], project_id=target_project_id
            )

            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=f"✅ Đã nạp thành công {num_chunks} đoạn văn bản từ '{info['filename']}' vào {scope_label}.",
            )
            logger.info(
                "Upload thành công: file=%s project_id=%s chunks=%s", info["filename"], target_project_id, num_chunks
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý callback upload: %s", exc)
            error_text = f"❌ Lỗi khi nạp tài liệu: {exc}"
            try:
                bot.edit_message_text(
                    chat_id=call.message.chat.id, message_id=call.message.message_id, text=error_text
                )
            except Exception:  # noqa: BLE001
                try:
                    bot.answer_callback_query(call.id, error_text[:200])
                except Exception:  # noqa: BLE001
                    pass

    # --------------------------------------------------------------------
    # Tin nhắn thường (text / ảnh) - luồng xử lý chính qua LangGraph
    # --------------------------------------------------------------------
    @bot.message_handler(content_types=["text", "photo"])
    def handle_message(message):
        try:
            is_group = message.chat.type in ("group", "supergroup")
            raw_text = message.text or message.caption or ""

            if is_group:
                is_command, remaining_text = _parse_group_command(raw_text, bot_username)
                is_reply_to_bot = (
                    message.reply_to_message is not None
                    and message.reply_to_message.from_user is not None
                    and message.reply_to_message.from_user.id == bot_user_id
                )
                if not (is_command or is_reply_to_bot):
                    # Tin nhắn chat thường giữa các operator, KHÔNG gọi bot -> bỏ qua
                    # hoàn toàn, tránh spam trả lời liên tục trong group.
                    return
                if is_command:
                    if not remaining_text.strip():
                        bot.reply_to(message, f"Gõ '{GROUP_COMMAND_PREFIX} <câu hỏi của bạn>' để hỏi bot.")
                        return
                    raw_text = remaining_text

            images = _extract_image_urls(bot, message)
            if not raw_text and not images:
                return  # bỏ qua tin nhắn rỗng (sticker, gif...)

            telegram_user_id = str(message.from_user.id) if message.from_user else ""
            group_id = str(message.chat.id)
            user_role = _resolve_role(telegram_user_id)

            from src.project_registry import get_project_id_for_group

            project_id = get_project_id_for_group(group_id)

            # Lệnh "dạy" AI (ADMIN only): 'LƯU LẠI: <nội dung kinh nghiệm>' - lưu thẳng
            # vào kho lịch sử sự cố qua RAG, KHÔNG đi qua LangGraph (đây là ghi dữ liệu,
            # không phải câu hỏi). Chỉ ADMIN mới được dùng - tránh OPERATOR/GUEST vô tình
            # (hoặc cố ý) ghi sai thông tin vào kho tham khảo chung, làm nhiễu RAG cho cả
            # nhà máy về sau (rủi ro dữ liệu, không phải rủi ro bảo mật, nhưng hậu quả vận
            # hành tương đương - 1 ghi chú sai có thể khiến AI tư vấn sai cho người khác).
            teach_prefix = next(
                (p for p in ADMIN_TEACH_PREFIXES if raw_text.strip().lower().startswith(p)), None
            )
            if teach_prefix is not None:
                if user_role != "ADMIN":
                    bot.reply_to(message, "⛔ Chỉ ADMIN (Kỹ sư Long) mới được dùng lệnh dạy 'LƯU LẠI:'.")
                    return
                note_text = raw_text.strip()[len(teach_prefix):].strip()
                if not note_text:
                    bot.reply_to(
                        message,
                        "Cú pháp: LƯU LẠI: <nội dung kinh nghiệm cần ghi nhớ>\n"
                        "Ví dụ: LƯU LẠI: sự cố lớp liệu quá dày ở ghi 2 >>> giảm Pause 10%, tăng gió cấp 1.",
                    )
                    return
                try:
                    from src.document_ingest import ingest_admin_note

                    point_id = ingest_admin_note(note_text, project_id=project_id)
                    logger.info("[handle_message] ADMIN đã dạy 1 ghi chú, id=%s, project_id=%s", point_id, project_id)
                    bot.reply_to(
                        message,
                        "✅ Đã lưu vào kho kinh nghiệm nội bộ. Lần sau có câu hỏi liên quan, "
                        "hệ thống sẽ tự trích dẫn ghi chú này.",
                    )
                except Exception as exc:  # noqa: BLE001 - không được để lệnh dạy làm chết bot
                    logger.exception("[handle_message] Lỗi lưu ghi chú ADMIN: %s", exc)
                    bot.reply_to(message, f"❌ Lưu ghi chú thất bại: {exc}\nVui lòng thử lại.")
                return

            initial_state = {
                "raw_message": raw_text,
                "user_role": user_role,
                "group_id": group_id,
                "project_id": project_id,
                "images": images,
                "loop_counter": 0,
                "messages": [],
                "routing_log": [],
            }

            bot.send_chat_action(message.chat.id, "typing")
            result = compiled_graph.invoke(initial_state)
            final_response = result.get("final_response", "Hệ thống không trả về phản hồi.")

            # Trả lời có thể vượt giới hạn 4096 ký tự của Telegram (câu trả lời phân tích
            # tài liệu dài) -> luôn dùng _send_long_reply để tự tách nhiều tin nhắn, tránh
            # lỗi "Bad Request: message is too long" làm mất trắng câu trả lời của người dùng.
            _send_long_reply(bot, message, final_response)
        except Exception as exc:  # noqa: BLE001 - không được để 1 tin nhắn lỗi làm chết bot
            logger.exception("Lỗi xử lý tin nhắn Telegram: %s", exc)
            try:
                error_text = f"❌ Đã xảy ra lỗi khi xử lý yêu cầu: {exc}\nVui lòng thử lại sau."
                _send_long_reply(bot, message, error_text)
            except Exception:  # noqa: BLE001
                pass  # nếu cả gửi lỗi cũng fail (vd mất mạng), bỏ qua, không crash bot

    return bot


def send_notification(group_id: str, text: str, image_data_urls: Optional[list[str]] = None,
                       is_emergency: bool = False) -> bool:
    """
    Gui 1 thong bao CHU DONG toi group_id (KHONG phai tra loi 1 tin nhan Telegram
    co san, khac voi _send_long_reply luon can 1 object `message` that de reply_to)
    - dung khi main.py (/invoke) nhan payload.notify_telegram=True hoac
    result["is_emergency"]=True, vi du: Boiler Station Agent goi /invoke truc tiep
    (khong qua webhook Telegram goc) va can day bao cao/canh bao ve group.

    LUU Y KIEN TRUC: day la lan dau /invoke can gui Telegram - truoc gio /invoke
    (main.py) chi tra JSON ve cho ben goi, khong bao gio dung toi Telegram; chi
    telegram_bot.py (qua webhook/long-polling that) moi tung gui Telegram. Ham nay
    dung LAI dung _bot_instance dang polling (khong tao TeleBot() moi, tranh 1 bot
    token ket noi 2 lan gay xung dot) - vi vay CHI hoat dong SAU KHI bot da ket noi
    xong (_bot_instance khac None); tai thoi diem app FastAPI vua khoi dong (vai
    giay dau, truoc khi polling thread ket noi xong), ham nay se fail-soft tra ve
    False, khong raise loi.

    image_data_urls: nhan dung dinh dang Station Agent dang gui - data URL kieu
    "data:image/png;base64,<...>" (xem screenshot.encode_image_base64() ben Station
    Agent) - tu giai ma base64 truoc khi goi bot.send_photo (Telegram API can bytes
    anh that, khong nhan duoc chuoi base64/data-URL truc tiep).

    is_emergency: neu True, dinh kem 1 nut inline "Da xac nhan" vao PHAN CUOI
    CUNG duoc gui (tin text cuoi neu co, khong thi anh cuoi) - ADMIN bam nut
    nay se tu dong chen lenh 'ack_alert' vao station_commands (xem
    handle_ack_alert_button ben tren), thay vi phai tu go SQL tay. Chi gan 1
    nut duy nhat cho ca lan gui (kha nang tin bi chia nhieu doan do qua dai),
    tranh nut bam bi lap lai nhieu lan gay roi mat.

    Fail-soft TUYET DOI: KHONG BAO GIO de loi Telegram lam fail request /invoke
    chinh (day la nguyen tac cong nghiep xuyen suot toan bo file nay) - moi loi deu
    duoc bat va log, ham chi tra ve True/False de bien goi (main.py) ghi log, khong
    anh huong gi den HTTP response tra ve cho Station Agent.
    """
    if _bot_instance is None:
        logger.warning(
            "[send_notification] Bot Telegram chua san sang (chua ket noi xong hoac "
            "dang trong chu ky reconnect) - bo qua gui thong bao chu dong toi "
            "group_id=%s. Neu loi nay lap lai nhieu lan, kiem tra TELEGRAM_BOT_TOKEN/"
            "TELEGRAM_POLL_ENABLED va log luc khoi dong app.", group_id,
        )
        return False

    try:
        chat_id = int(group_id)
    except (TypeError, ValueError):
        logger.warning("[send_notification] group_id không hợp lệ để gửi Telegram: %r", group_id)
        return False

    ok = True

    markup = None
    if is_emergency:
        from telebot import types

        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton(
                "✅ Đã xác nhận (tạm ngưng cảnh báo lặp lại)", callback_data=f"ackalert:{group_id}"
            )
        )

    text_chunks = _split_long_message(text) if text else []
    images = image_data_urls or []
    total_parts = len(text_chunks) + len(images)
    part_idx = 0

    for chunk in text_chunks:
        part_idx += 1
        part_markup = markup if part_idx == total_parts else None
        sent = False
        for attempt in range(1, 4):
            try:
                _bot_instance.send_message(chat_id, chunk, reply_markup=part_markup, timeout=60)
                sent = True
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[send_notification] Lỗi gửi text lần %s/3 (chat_id=%s): %s",
                    attempt, chat_id, exc,
                )
                if attempt < 3:
                    time.sleep(RETRY_SLEEP_SECONDS)
        if not sent:
            ok = False

    for data_url in images:
        part_idx += 1
        part_markup = markup if part_idx == total_parts else None
        try:
            b64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw_bytes = base64.b64decode(b64_part)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[send_notification] Lỗi giải mã ảnh base64 (bỏ qua ảnh này): %s", exc)
            ok = False
            continue

        sent = False
        for attempt in range(1, 4):
            try:
                _bot_instance.send_photo(chat_id, io.BytesIO(raw_bytes), reply_markup=part_markup, timeout=90)
                sent = True
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[send_notification] Lỗi gửi ảnh lần %s/3 (chat_id=%s): %s",
                    attempt, chat_id, exc,
                )
                if attempt < 3:
                    time.sleep(RETRY_SLEEP_SECONDS)
        if not sent:
            ok = False

    if ok:
        logger.info("[send_notification] Đã gửi thông báo chủ động tới group_id=%s.", group_id)
    return ok


def _polling_loop(compiled_graph):
    """
    Vòng lặp long-polling chính, tự động reconnect khi mất mạng. Chạy ở
    background thread (daemon) để không chặn FastAPI event loop chính.

    QUAN TRỌNG - đã từng có lỗi thực tế: dùng bot.infinity_polling(...) (vốn được
    quảng cáo là tự retry vô hạn) nhưng khi gặp lỗi 409 Conflict (thường chỉ là
    xung đột NGẮN HẠN lúc Render đang chuyển từ instance cũ sang instance mới khi
    deploy/restart), thư viện telebot lại ÂM THẦM DỪNG HẲN vòng polling nội bộ và
    KHÔNG BAO GIỜ tự thử lại - khiến bot "chết" vĩnh viễn cho tới khi có người vào
    Render bấm Restart tay. Để không bao giờ phụ thuộc vào cơ chế retry nội bộ của
    thư viện (vốn không đáng tin cậy ở đây), dùng bot.polling() (KHÔNG dùng
    infinity_polling) và tự đảm bảo luôn sleep + quay lại đầu vòng lặp NGOÀI CÙNG
    dù bot.polling() kết thúc theo cách nào (raise exception hay tự return) -
    dùng try/finally thay vì chỉ try/except để đảm bảo chắc chắn 100% job này.
    """
    while True:
        try:
            bot = _build_bot(compiled_graph)
            global _bot_instance
            _bot_instance = bot
            logger.info("Telegram bot bắt đầu long-polling.")
            bot.polling(non_stop=False, interval=1, timeout=30, long_polling_timeout=30)
            logger.warning(
                "Vòng polling của Telegram bot đã tự kết thúc (không phải do lỗi) - "
                "sẽ khởi động lại sau %ss.",
                RETRY_SLEEP_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - bắt mọi lỗi để vòng lặp không bao giờ chết hẳn
            logger.error(
                "Telegram bot gặp lỗi (mất mạng / token sai / xung đột 409 tạm thời / "
                "Telegram API down): %s. Ngủ đông %ss rồi thử kết nối lại.",
                exc,
                RETRY_SLEEP_SECONDS,
            )
        finally:
            # LUÔN sleep + quay lại vòng lặp, bất kể bot.polling() thoát ra vì lý do
            # gì - đây là điểm mấu chốt khắc phục lỗi "bot chết im lặng" đã gặp.
            time.sleep(RETRY_SLEEP_SECONDS)


def start_telegram_bot_background(compiled_graph) -> Optional[threading.Thread]:
    """
    Khởi động bot Telegram trên 1 daemon thread riêng, gọi từ FastAPI startup
    event trong main.py. Nếu TELEGRAM_POLL_ENABLED=false hoặc token chưa cấu
    hình, bỏ qua không khởi động (vẫn cho phép dùng server chỉ qua API /invoke).
    """
    global _bot_thread

    enabled = os.getenv("TELEGRAM_POLL_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")

    if not enabled:
        logger.info("TELEGRAM_POLL_ENABLED=false, bỏ qua khởi động Telegram bot.")
        return None
    if not token or token.startswith("xxxxxx"):
        logger.warning("TELEGRAM_BOT_TOKEN chưa được cấu hình, bỏ qua khởi động Telegram bot.")
        return None

    _bot_thread = threading.Thread(
        target=_polling_loop, args=(compiled_graph,), daemon=True, name="telegram-bot-polling"
    )
    _bot_thread.start()
    logger.info("Đã khởi động Telegram bot polling thread.")
    return _bot_thread
