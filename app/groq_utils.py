import os
from functools import lru_cache
from typing import Iterable, List, Sequence

from groq import Groq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@lru_cache(maxsize=1)
def _client() -> Groq:
    """Create a cached Groq client using `GROQ_API_KEY` environment variable."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set but Groq model was requested.")
    return Groq(api_key=GROQ_API_KEY)


def groq_chat_completion(
    model: str,
    messages: Sequence[dict],
    temperature: float = 0.85,
    max_tokens: int = 400,
    top_p: float = 1.0,
    stream: bool = False,
) -> str | Iterable[str]:
    """Run a Groq chat completion and return text or a stream of chunks."""
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
    )

    if stream:
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
    else:
        return response.choices[0].message.content.strip()


"""Groq client helpers.

Thin wrappers around Groq chat completions to match internal calling style
and allow optional streaming.
"""
