# Twilio-OpenAI-WhatsApp-Bot/app/openai_utils.py

import os 
from dotenv import load_dotenv
from litellm import completion
from app.prompts import SUMMARY_PROMPT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# IF YOU WANT TO ADD MORE MODELS
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# REGION_NAME = os.getenv("REGION_NAME")

# os.environ['GROQ_API_KEY'] = GROQ_API_KEY
# os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
# os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
# os.environ["AWS_REGION_NAME"] = REGION_NAME

# Constants
TEMPERATURE = 0.35
MAX_TOKENS = 350
STOP_SEQUENCES = ["==="]
TOP_P = 0.9
TOP_K = 0.1
BEST_OF = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0


SUPPORTED_MODELS = {
    # Groq Llama models
    "groq/llama3-8b-8192", 
    "groq/llama-3.1-8b-instant", 
    "groq/llama-3.1-70b-versatile", 
    # OpenAI models
    "gpt-3.5-turbo-0125",
    "gpt-4o", 
    "gpt-4o-mini",
    "gpt-4-0125-preview",
    # Amazon Anthropic models
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/anthropic.claude-v2:1",
    }


def gpt_without_functions(model, stream=False, messages=[]):
    """ GPT model without function call. """
    if model not in SUPPORTED_MODELS:
        return False
    response = completion(
        model=model, 
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stream=stream
    )
    return response 



def summarise_conversation(history):
    """Summarise conversation history in one sentence"""

    if not history:
        return "No prior conversation."

    conversation_lines = []
    for item in history[-70:]:
        role = item.get("role")
        content = item.get("content")
        if role and content:
            conversation_lines.append(f"{role.capitalize()}: {content}")
            continue
        if "user_input" in item:
            conversation_lines.append(f"User: {item['user_input']}")
        if "bot_response" in item:
            conversation_lines.append(f"Assistant: {item['bot_response']}")

    if not conversation_lines:
        return "No prior conversation."

    conversation = "\n".join(conversation_lines)

    openai_response = gpt_without_functions(
                        model="gpt-3.5-turbo-0125",
                        stream=False,
                        messages=[
                            {'role': 'system', 'content': SUMMARY_PROMPT}, 
                            {'role': 'user', 'content': conversation}
                    ])
    chatbot_response = openai_response.choices[0].message.content.strip()

    return chatbot_response


def translate_text_to_urdu(text: str) -> str:
    try:
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "global")
        if project_id:
            from google.cloud import translate
            client = translate.TranslationServiceClient()
            parent = f"projects/{project_id}/locations/{location}"
            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [text],
                    "mime_type": "text/plain",
                    "target_language_code": "ur",
                }
            )
            translated = response.translations[0].translated_text
            if translated:
                return translated.strip()
    except Exception:
        pass
    return text


def detect_language(text: str) -> str | None:
    try:
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "global")
        if project_id:
            from google.cloud import translate
            client = translate.TranslationServiceClient()
            parent = f"projects/{project_id}/locations/{location}"
            response = client.detect_language(
                request={
                    "parent": parent,
                    "content": text,
                }
            )
            if response.languages:
                code = response.languages[0].language_code
                if code:
                    code = code.split("-")[0].lower()
                return code
    except Exception:
        pass
    return None


def translate_text(text: str, target_language_code: str) -> str:
    if not text:
        return ""
    try:
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "global")
        if project_id:
            from google.cloud import translate
            client = translate.TranslationServiceClient()
            parent = f"projects/{project_id}/locations/{location}"
            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [text],
                    "mime_type": "text/plain",
                    "target_language_code": target_language_code,
                }
            )
            translated = response.translations[0].translated_text
            if translated:
                return translated.strip()
    except Exception:
        pass
    return text


def detect_and_translate_to_english(text: str) -> tuple[str, str | None]:
    code = detect_language(text) or "en"
    if code == "en":
        return text, code
    translated = translate_text(text, "en")
    return translated, code
