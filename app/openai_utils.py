"""Language utilities for PSER chatbot.

Provides:
- Conversation summarization for short memory
- Language detection (Google Cloud Translate v3)
- Text translation (Google Cloud Translate v3)
- Thin wrapper for LLM calls used in summarization

These helpers are used by the FastAPI handlers to normalize user input and
produce answers in the user's original language.
"""

import os 
from dotenv import load_dotenv
from litellm import completion
from app.prompts import SUMMARY_PROMPT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
USE_GPT_DETECTION = os.getenv("USE_GPT_DETECTION", "false").lower() == "true"
USE_GPT_FORWARD_TRANSLATION = os.getenv("USE_GPT_FORWARD_TRANSLATION", "false").lower() == "true"

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
    """Call an LLM for non-tool usage (used for summarization)."""
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
    """Summarize recent conversation turns into a short, single paragraph."""

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
    """Translate a text snippet to Urdu using Google Cloud Translate.

    Returns the original text if translation is unavailable.
    """
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
    """Detect the ISO 639-1 language code using Google Cloud Translate.

    Returns `None` if detection is unavailable.
    """
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
    """Translate text to a target language code (ISO 639-1) using Google.

    Returns the input text unchanged when translation is unavailable.
    """
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
    """Detect language and return an English-normalized text copy plus code."""
    code = detect_language(text) or "ur"
    if code == "en":
        return text, code
    if USE_GPT_FORWARD_TRANSLATION:
        try:
            resp = completion(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the user's message into English. Preserve meaning and entities. Return only the translated text.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                stream=False,
            )
            translated = resp.choices[0].message.content.strip()
        except Exception:
            translated = translate_text(text, "en")
        return translated, code
    translated = translate_text(text, "en")
    return translated, code

def translate_back_to_source(text: str, source_language_code: str | None) -> str:
    if not text:
        return ""
    if not source_language_code or source_language_code == "en":
        return text
    return translate_text(text, source_language_code)

def translate_back_to_source_gpt(
    question: str,
    answer: str,
    source_language_code: str | None,
    model: str = "gpt-4o",
) -> str:
    if not answer:
        return ""
    if not source_language_code or source_language_code == "en":
        return answer
    messages = [
        {
            "role": "system",
            "content": (
                "Translate the provided Answer into the same language as the provided Question. "
                "Do not add, remove, explain, paraphrase, or change meaning. "
                "Preserve names, entities, formatting, and keep the phone number '0800-02345' unchanged. "
                "If the Answer contains 'Final Answer:', keep that phrase unchanged and translate only the remainder. "
                "Return only the translated answer text in the same language as the Question."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nAnswer:\n{answer}",
        },
    ]
    try:
        resp = completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            stream=False,
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            return translate_text(answer, source_language_code)
        return out
    except Exception:
        return translate_text(answer, source_language_code)

def final_answer_label(source_language_code: str | None, model: str = "gpt-4o") -> str:
    code = (source_language_code or "en").lower()
    mapping = {
        "en": "Final Answer:",
        "ur": "حتمی جواب:",
        "es": "Respuesta final:",
        "hi": "अंतिम उत्तर:",
        "ar": "الإجابة النهائية:",
        "pa": "آخری جواب:",
        "fr": "Réponse finale :",
        "de": "Endgültige Antwort:",
        "tr": "Nihai cevap:",
    }
    if code in mapping:
        return mapping[code]
    try:
        resp = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the phrase 'Final Answer:' into the language with ISO 639-1 code '{code}'. Return only the translated phrase.",
                },
                {"role": "user", "content": "Final Answer:"},
            ],
            temperature=0.0,
            max_tokens=16,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            stream=False,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or "Final Answer:"
    except Exception:
        return "Final Answer:"

