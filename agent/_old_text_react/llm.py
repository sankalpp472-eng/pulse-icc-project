"""
LLM client — Groq by default, switchable to Ollama via LLM_BACKEND env var.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()


def _get_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to agent/.env")
    return Groq(api_key=api_key)


def chat(messages: list[dict], model: str = None) -> str:
    """
    Send a list of messages and return the assistant's reply text.

    messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    if _BACKEND == "groq":
        return _groq_chat(messages, model or "qwen/qwen3.8-27b")
    elif _BACKEND == "ollama":
        return _ollama_chat(messages, model or os.getenv("OLLAMA_MODEL", "llama3.1"))
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {_BACKEND!r}. Use 'groq' or 'ollama'.")


def _groq_chat(messages: list[dict], model: str) -> str:
    import time
    client = _get_groq_client()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"[llm] Rate limit hit, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq rate limit: max retries exceeded.")


def _ollama_chat(messages: list[dict], model: str) -> str:
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
