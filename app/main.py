import os
import json
from dotenv import load_dotenv

from fastapi import Form, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from twilio.rest import Client

from app.cookies_utils import set_cookies, get_cookies, clear_cookies
from app.openai_utils import summarise_conversation
from app.redis_utils import redis_conn
from app.logger_utils import logger
from app.rag_utils import answer_question as answer_with_rag

# Load environment variables from a .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

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
    """ Send a message via Twilio WhatsApp """
    TWILIO_WHATSAPP_PHONE_NUMBER = "whatsapp:" + TWILIO_WHATSAPP_NUMBER
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    from_whatsapp_number = TWILIO_WHATSAPP_PHONE_NUMBER
    twilio_client.messages.create(body=message,
                                  from_=from_whatsapp_number,
                                  to=to_number)


@app.post('/whatsapp-endpoint')
async def whatsapp_endpoint(request: Request, From: str = Form(...), Body: str = Form(...)):
    logger.info(f'WhatsApp endpoint triggered...')
    logger.info(f'Request: {request}')
    logger.info(f'From: {From}')
    logger.info(f'Incoming message: {Body}')

    query = Body
    phone_no = From.replace('whatsapp:+', '')
    chat_session_id = phone_no

    # Retrieve chat history from Redis
    history = get_cookies(redis_conn, f'whatsapp_twilio_demo_{chat_session_id}_history') or []
    if history:
        history = json.loads(history)
    
    # Append the user's query to the chat history
    history.append({"role": 'user', "content": query})

    # Summarize the conversation history
    history_summary = summarise_conversation(history)

    # Answer user query using the RAG pipeline
    rag_response = answer_with_rag(query, history_summary=history_summary, chat_history=history)

    def extract_final_answer(response_text: str) -> str:
        marker = "Final Answer:"
        if marker in response_text:
            return response_text.split(marker, 1)[1].strip()
        return response_text.strip()

    chatbot_response = extract_final_answer(rag_response)
    logger.info(f'Outgoing response: {chatbot_response}')

    # Append the assistant's response to the chat history on Redis
    history.append(
        {
            'role': 'assistant',
            'content': chatbot_response,
            'raw_response': rag_response,
        }
    )
    set_cookies(redis_conn, name=f'whatsapp_twilio_demo_{chat_session_id}_history', value=json.dumps(history))

    # Send the assistant's response back to the user via WhatsApp
    respond(From, chatbot_response)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host='0.0.0.0', port=3002, reload=True)