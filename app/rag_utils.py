"""RAG utilities for PSER chatbot.

This module builds a simple retrieval-augmented generation pipeline:
- Loads a plain-text knowledge base from `DATA_RAG` or the default PSER file
- Splits text into chunks and indexes them in a FAISS vector store with OpenAI embeddings
- Formats recent chat history and optional summary
- Invokes an LLM with a system prompt tailored to PSER guidelines

Functions here are cached where appropriate to reduce startup and runtime overhead.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "PSER_info.txt"
DATA_RAG_ENV = os.getenv("DATA_RAG")
DATA_PATH = Path(DATA_RAG_ENV).expanduser() if DATA_RAG_ENV else DEFAULT_DATA_PATH
CONTEXTUALIZER_ENABLED = os.getenv("ENABLE_CONTEXTUALIZER", "false").lower() == "true"
DEFAULT_HISTORY_SUMMARY = "The user has just started the conversation."
MAX_HISTORY_MESSAGES = 4


def _load_corpus() -> str:
    """Read the knowledge base .txt file into a single string.

    Raises FileNotFoundError if the path does not exist and ValueError for non-.txt files.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {DATA_PATH}")
    if DATA_PATH.suffix.lower() != ".txt":
        raise ValueError(f"Knowledge base must point to a .txt file, got {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _vectorstore() -> FAISS:
    """Build and cache the FAISS vector store from the loaded corpus."""
    corpus = _load_corpus()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(corpus)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(documents, embedding=embeddings)


@lru_cache(maxsize=1)
def _rag_chain():
    """Create and cache the PSER RAG prompt chain with the chat LLM."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.85)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Role: You are a chatbot for the Punjab Socio-Economic Registry (PSER), an initiative by the Punjab Government.\n\n"
                    "Instructions:\n"
                    "- Answer only using information in the supplied context ({context}).\n"
                    "- Do not use external knowledge or make assumptions.\n"
                    "- If the answer is not found in the context, reply: \"please contact at 0800-02345.\"\n"
                    "- Always display the helpline number as: 0800-02345\n"
                    "- Reference {history_summary} and {chat_history} as needed to inform responses.\n\n"
                    "Language Requirement:\n"
                    "- IMPORTANT: You must respond in the EXACT SAME LANGUAGE as the user's question.\n"
                    "- If the user asks in Urdu, respond in Urdu. If they ask in English, respond in English.\n"
                    "- If they ask in any other language, respond in that same language.\n"
                    "- Do NOT translate the response. Keep the response in the original language of the question.\n\n"
                    "Response Guidelines:\n"
                    "- Keep answers concise (maximum 120 words).\n"
                    "- Start each reply with: Final Answer: <answer>\n"
                    "- The \"Final Answer:\" label should also be in the same language as the user's question.\n"
                    "- For Urdu, use: حتمی جواب:\n"
                    "- For English, use: Final Answer:\n"
                    "- For other languages, translate \"Final Answer:\" appropriately to that language.\n"
                    "(Example in English: Final Answer: The registration period for PSER is March–April. Please contact at 0800-02345 for further details.)\n"
                    "(Example in Urdu: حتمی جواب: PSER کی رجسٹریشن کی مدت مارچ-اپریل ہے۔ مزید تفصیلات کے لیے برائے کرم 0800-02345 پر رابطہ کریں۔)\n\n"
                    "Escalation:\n"
                    "- If the context lacks the answer, instruct the user to contact the helpline."
                    "- If Greeted respond by introducing yourself and ask how can I help regarding PSER Querries."
                ),
            ),
            ("human", "{question}"),
        ]
    )
    return prompt | llm


@lru_cache(maxsize=1)
def _contextualizer_chain():
    """Create and cache a light contextualizer to rewrite follow-ups to standalone questions."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Rewrite follow-up questions into standalone questions using the "
                    "conversation provided. Keep the meaning but resolve references. "
                    "If no rewrite is needed, return the original question. "
                    "Return only the question text."
                ),
            ),
            (
                "human",
                (
                    "Conversation so far:\n{chat_history}\n\n"
                    "Follow-up question: {question}\n\n"
                    "Standalone question:"
                ),
            ),
        ]
    )
    return prompt | llm


def _build_context(docs: List[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _format_chat_history(history: Sequence[dict] | None) -> str:
    """Return the most recent `MAX_HISTORY_MESSAGES` turns as human-readable lines."""
    if not history:
        return "No previous turns."
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    formatted = []
    for message in trimmed:
        role = message.get("role", "assistant").capitalize()
        content = message.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def _contextualize_question(question: str, chat_history: Sequence[dict] | None) -> str:
    """Optionally rewrite a follow-up into a standalone question using recent history."""
    if not CONTEXTUALIZER_ENABLED or not chat_history:
        return question
    chain = _contextualizer_chain()
    response = chain.invoke(
        {
            "chat_history": _format_chat_history(chat_history),
            "question": question,
        }
    )
    rewritten = response.content.strip()
    return rewritten or question


def answer_question(
    question: str,
    history_summary: str | None = None,
    chat_history: Sequence[dict] | None = None,
) -> str:
    """Retrieve relevant PSER context and answer the user question.

    This function:
    - Optionally rewrites the question using recent chat history
    - Performs a similarity search over the FAISS index
    - Builds the combined context and invokes the RAG chain
    - Returns the LLM output; the caller is expected to extract the
      `Final Answer:` portion for display
    """
    if not question:
        return "Final Answer: I didn't receive a question to answer."

    search_query = _contextualize_question(question, chat_history)

    vectorstore = _vectorstore()
    docs = vectorstore.similarity_search(search_query, k=3)
    context = _build_context(docs)
    chain = _rag_chain()
    response = chain.invoke(
        {
            "context": context,
            "history_summary": history_summary or DEFAULT_HISTORY_SUMMARY,
            "chat_history": _format_chat_history(chat_history),
            "question": question,
        }
    )
    return response.content.strip()

