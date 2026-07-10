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

import logging
import os
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger("telegram_bot")

ADMIN_ID = os.getenv("ADMIN_ID", "")
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


def _send_long_reply(bot, message, text: str) -> None:
    """Gửi trả lời có thể dài hơn giới hạn Telegram - tự tách thành nhiều tin nhắn liên tiếp."""
    chunks = _split_long_message(text)
    bot.reply_to(message, chunks[0])
    for chunk in chunks[1:]:
        bot.send_message(message.chat.id, chunk)


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
    from telebot import types

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
            "• /list_docs - xem danh sách tài liệu đã nạp cho dự án này.\n"
            "• Gửi file .txt/.pdf/.docx - (ADMIN) nạp tài liệu vào kho kiến thức.",
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

            upload_id = uuid.uuid4().hex[:16]
            with _pending_uploads_lock:
                _pending_uploads[upload_id] = {
                    "file_id": message.document.file_id,
                    "filename": filename,
                    "group_id": str(message.chat.id),
                    "telegram_user_id": telegram_user_id,
                }

            from src.project_registry import get_project_id_for_group

            current_project = get_project_id_for_group(str(message.chat.id))

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
            bot.reply_to(
                message,
                f"📄 Đã nhận file '{filename}'. Chọn nơi lưu tài liệu:",
                reply_markup=markup,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi xử lý nhận file upload: %s", exc)
            try:
                bot.reply_to(message, f"❌ Lỗi khi nhận file: {exc}")
            except Exception:  # noqa: BLE001
                pass

    @bot.callback_query_handler(func=lambda call: call.data.startswith(("kbshared:", "kbproject:")))
    def handle_upload_choice(call):
        try:
            scope, upload_id = call.data.split(":", 1)
            with _pending_uploads_lock:
                info = _pending_uploads.pop(upload_id, None)

            if not info:
                bot.answer_callback_query(call.id, "Phiên upload đã hết hạn (bot có thể đã khởi động lại). Gửi lại file.")
                return

            if _resolve_role(info["telegram_user_id"]) != "ADMIN":
                bot.answer_callback_query(call.id, "Chỉ ADMIN mới được thao tác.")
                return

            bot.answer_callback_query(call.id, "Đang xử lý...")
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=f"⏳ Đang nạp '{info['filename']}'... (thường mất 10-30 giây, vui lòng đợi)",
            )

            from src.document_ingest import ingest_document
            from src.project_registry import SHARED_PROJECT_ID, get_project_id_for_group

            if scope == "kbshared":
                target_project_id = SHARED_PROJECT_ID
                scope_label = "kho DÙNG CHUNG"
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


def _polling_loop(compiled_graph):
    """
    Vòng lặp long-polling chính, tự động reconnect khi mất mạng. Chạy ở
    background thread (daemon) để không chặn FastAPI event loop chính.
    """
    while True:
        try:
            bot = _build_bot(compiled_graph)
            global _bot_instance
            _bot_instance = bot
            logger.info("Telegram bot bắt đầu long-polling.")
            bot.infinity_polling(timeout=30, long_polling_timeout=30)
        except Exception as exc:  # noqa: BLE001 - bắt mọi lỗi để vòng lặp không bao giờ chết hẳn
            logger.error(
                "Telegram bot gặp lỗi (mất mạng / token sai / Telegram API down): %s. "
                "Ngủ đông %ss rồi thử kết nối lại.",
                exc,
                RETRY_SLEEP_SECONDS,
            )
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
