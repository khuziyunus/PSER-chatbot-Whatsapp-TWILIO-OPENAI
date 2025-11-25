import os
from functools import lru_cache
from typing import Iterable, List, Sequence

from groq import Groq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@lru_cache(maxsize=1)
def _client() -> Groq:
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
    """
    Thin wrapper around Groq chat completions that matches our internal calling style.
    Returns either the full response text or yields streamed chunks depending on `stream`.
    """
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

