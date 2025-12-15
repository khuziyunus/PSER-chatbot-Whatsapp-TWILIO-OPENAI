import os
import json
from dotenv import load_dotenv

from fastapi import Form, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from twilio.rest import Client

from app.cookies_utils import set_cookies, get_cookies, clear_cookies
from app.openai_utils import summarise_conversation, translate_text_to_urdu
from app.redis_utils import redis_conn
from app.logger_utils import logger
from app.rag_utils import answer_question as answer_with_rag
from app.whatsapp_service import process_whatsapp_message

# Load environment variables from a .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
ENABLE_CHAT_HISTORY = os.getenv("ENABLE_CHAT_HISTORY", "true").lower() == "true"

app = FastAPI(
    title="PSER-Twilio-OpenAI-WhatsApp-Bot",
    description="PSER WhatsApp Bot",
    version="0.0.2",

)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

def respond(to_number, message) -> None:
    """Send a WhatsApp message via Twilio."""
    TWILIO_WHATSAPP_PHONE_NUMBER = "whatsapp:" + TWILIO_WHATSAPP_NUMBER
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    from_whatsapp_number = TWILIO_WHATSAPP_PHONE_NUMBER
    twilio_client.messages.create(body=message,
                                  from_=from_whatsapp_number,
                                  to=to_number)


@app.post('/whatsapp-endpoint')
async def whatsapp_endpoint(request: Request, From: str = Form(...), Body: str = Form(...)):
    """Handle inbound WhatsApp messages from Twilio.

    - Logs inbound request
    - Detects and normalizes language, translates to English for RAG
    - Runs RAG, extracts `Final Answer`, translates back to source language
    - Persists minimal chat history in Redis (if enabled)
    - Sends response via Twilio
    """
    logger.info(f'WhatsApp endpoint triggered...')
    logger.info(f'Request: {request}')
    logger.info(f'From: {From}')
    logger.info(f'Incoming message: {Body}')

    query = Body
    query_urdu = translate_text_to_urdu(query)
    phone_no = From.replace('whatsapp:+', '')
    chat_session_id = phone_no

    history = []
    history_summary = "Chat history disabled."
    chat_history_for_rag = None

    if ENABLE_CHAT_HISTORY:
        stored_history = get_cookies(redis_conn, f'whatsapp_twilio_demo_{chat_session_id}_history') or []
        if stored_history:
            history = json.loads(stored_history)
        history.append({"role": 'user', "content": query})
        history_summary = summarise_conversation(history)
        chat_history_for_rag = history

    # Answer user query using the RAG pipeline
    rag_response = answer_with_rag(query_urdu, history_summary=history_summary, chat_history=chat_history_for_rag)

    def extract_final_answer(response_text: str) -> str:
        marker = "Final Answer:"
        if marker in response_text:
            return response_text.split(marker, 1)[1].strip()
        return response_text.strip()

    chatbot_response = extract_final_answer(rag_response)
    chatbot_response_urdu = translate_text_to_urdu(chatbot_response)
    logger.info(f'Outgoing response: {chatbot_response_urdu}')

    if ENABLE_CHAT_HISTORY:
        history.append(
            {
                'role': 'assistant',
                'content': chatbot_response_urdu,
                'raw_response': rag_response,
            }
        )
        set_cookies(redis_conn, name=f'whatsapp_twilio_demo_{chat_session_id}_history', value=json.dumps(history))

    # Send the assistant's response back to the user via WhatsApp
    respond(From, chatbot_response_urdu)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host='0.0.0.0', port=3002, reload=True)
