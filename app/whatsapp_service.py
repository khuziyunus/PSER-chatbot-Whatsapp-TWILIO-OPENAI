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
    """Main WhatsApp message entry: answer with RAG in same language, store."""
    history: list[dict] = []
    history_summary = "Chat history disabled."
    chat_history_for_rag = None
    query = message or ""
    # Pass original question directly to RAG - model will respond in same language
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
    # RAG chain now handles responding in the same language as the question
    rag_response = _safe_answer_with_rag(query, history_summary=history_summary, chat_history=chat_history_for_rag)
    # The response already includes "Final Answer:" in the appropriate language
    final_response = rag_response
    logger.info(f"Outgoing response: {final_response}")
    if enable_history and session_id:
        history.append({"role": "assistant", "content": final_response, "raw_response": rag_response})
        _save_history(session_id, history)
    return final_response
"""WhatsApp message processing helpers.

Process WhatsApp messages through RAG pipeline with chat history storage.
The RAG model responds in the same language as the user's question.
"""
