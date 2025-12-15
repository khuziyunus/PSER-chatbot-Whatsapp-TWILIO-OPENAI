import json
from typing import Sequence

from app.logger_utils import logger
from app.cookies_utils import set_cookies, get_cookies
from app.redis_utils import redis_conn


def _extract_final_answer(response_text: str) -> str:
    """Extract the content after the `Final Answer:` marker if present."""
    marker = "Final Answer:"
    if marker in response_text:
        return response_text.split(marker, 1)[1].strip()
    return response_text.strip()


def _safe_detect_translate_to_english(text: str) -> tuple[str, str | None]:
    """Detect language and translate to English, falling back to identity."""
    try:
        from app.openai_utils import detect_and_translate_to_english
        return detect_and_translate_to_english(text)
    except Exception:
        return text, "en"


def _safe_translate(text: str, target_language_code: str | None, question: str | None = None) -> str:
    """Translate `text` to `target_language_code` with safe fallbacks and context."""
    if not text:
        return ""
    if not target_language_code or target_language_code == "en":
        return text
    try:
        from app.openai_utils import translate_back_to_source_gpt
        return translate_back_to_source_gpt(question or "", text, target_language_code)
    except Exception:
        return text


def _safe_answer_with_rag(question: str, history_summary: str | None, chat_history: Sequence[dict] | None) -> str:
    """Run the RAG pipeline and return the model output; fallback on errors."""
    try:
        from app.rag_utils import answer_question as answer_with_rag
        return answer_with_rag(question, history_summary=history_summary, chat_history=chat_history)
    except Exception:
        return "Final Answer: Please contact at [insert helpline]"


def _load_history(session_id: str) -> list[dict]:
    """Load serialized chat history for a session from Redis."""
    try:
        stored = get_cookies(redis_conn, f"whatsapp_twilio_demo_{session_id}_history") or []
        if isinstance(stored, str):
            return json.loads(stored)
        return stored or []
    except Exception:
        return []


def _save_history(session_id: str, history: list[dict]) -> None:
    """Persist chat history for a session into Redis."""
    try:
        set_cookies(redis_conn, name=f"whatsapp_twilio_demo_{session_id}_history", value=json.dumps(history))
    except Exception:
        pass


def process_whatsapp_message(message: str, session_id: str | None, enable_history: bool) -> str:
    """Main WhatsApp message entry: normalize, answer with RAG, translate, store."""
    history: list[dict] = []
    history_summary = "Chat history disabled."
    chat_history_for_rag = None
    query = message or ""
    query_english, source_lang = _safe_detect_translate_to_english(query)
    if enable_history and session_id:
        history = _load_history(session_id)
        history.append({"role": "user", "content": query})
        try:
            from app.openai_utils import summarise_conversation
            history_summary = summarise_conversation(history)
            chat_history_for_rag = history
        except Exception:
            history_summary = "Chat history disabled."
            chat_history_for_rag = None
    rag_response = _safe_answer_with_rag(query_english, history_summary=history_summary, chat_history=chat_history_for_rag)
    chatbot_response = _extract_final_answer(rag_response)
    final_response = _safe_translate(chatbot_response, source_lang, query)
    logger.info(f"Outgoing response: {final_response}")
    if enable_history and session_id:
        history.append({"role": "assistant", "content": final_response, "raw_response": rag_response})
        _save_history(session_id, history)
    return final_response
"""WhatsApp message processing helpers.

Wrap language normalization, RAG answering, translation, and chat history
storage with defensive error handling for production resilience.
"""
