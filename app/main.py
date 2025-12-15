import os
import json
from dotenv import load_dotenv

from fastapi import Form, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from twilio.rest import Client

from app.cookies_utils import set_cookies, get_cookies, clear_cookies
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
    phone_no = From.replace('whatsapp:+', '')
    chat_session_id = phone_no

    final_response = process_whatsapp_message(message=query, session_id=chat_session_id, enable_history=ENABLE_CHAT_HISTORY)
    logger.info(f'Outgoing response: {final_response}')

    # Send the assistant's response back to the user via WhatsApp
    respond(From, final_response)

