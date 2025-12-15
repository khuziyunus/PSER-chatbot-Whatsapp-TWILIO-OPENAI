from app.logger_utils import logger


def _extract_final_answer(response_text: str) -> str:
    """Extract the answer portion after the `Final Answer:` marker."""
    marker = "Final Answer:"
    if marker in response_text:
        return response_text.split(marker, 1)[1].strip()
    return response_text.strip()


def process_web_message(message: str) -> str:
    """Process a plain text web message through the RAG pipeline."""
    query = message or ""
    try:
        from app.openai_utils import detect_and_translate_to_english, translate_back_to_source_gpt
        from app.rag_utils import answer_question as answer_with_rag
        query_english, source_lang = detect_and_translate_to_english(query)
        rag_response = answer_with_rag(query_english, history_summary="Chat history disabled.", chat_history=None)
        chatbot_response = _extract_final_answer(rag_response)
        final_response = translate_back_to_source_gpt(query, chatbot_response, source_lang)
        return final_response
    except Exception:
        return "Please contact at [insert helpline]"
"""Web chat message processing helpers.

Mirrors the WhatsApp flow for a web UI: detect/translate, run RAG,
extract `Final Answer`, and translate back to the user's language.
"""
