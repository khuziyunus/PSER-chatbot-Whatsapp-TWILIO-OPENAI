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


DATA_PATH = Path(__file__).resolve().parent / "data" / "acm_giki.txt"
DEFAULT_HISTORY_SUMMARY = "The user has just started the conversation."
MAX_HISTORY_MESSAGES = 8

load_dotenv()


def _load_corpus() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _vectorstore() -> FAISS:
    corpus = _load_corpus()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(corpus)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(documents, embedding=embeddings)


@lru_cache(maxsize=1)
def _rag_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.85)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant for the ACM GIKI Chapter. "
                    "Use the provided context to answer the user's question. "
                    "If the answer is not contained in the context, say you do not know. "
                    "Previous conversation summary: {history_summary}\n\n"
                    "Recent conversation turns:\n{chat_history}\n\n"
                    "Context:\n{context}\n\n"
                    "Respond concisely and keep answers under 120 words. "
                    "Format your response exactly as:\n"
                    "Final Answer: <answer>"
                ),
            ),
            ("human", "{question}"),
        ]
    )
    return prompt | llm


@lru_cache(maxsize=1)
def _contextualizer_chain():
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
    return "\n\n".join(doc.page_content for doc in docs)


def _format_chat_history(history: Sequence[dict] | None) -> str:
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
    if not chat_history:
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
    """
    Retrieve relevant context from the ACM GIKI corpus and answer a user question.
    """
    if not question:
        return "Final Answer: I didn't receive a question to answer."

    search_query = _contextualize_question(question, chat_history)

    vectorstore = _vectorstore()
    docs = vectorstore.similarity_search(search_query, k=5)
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

