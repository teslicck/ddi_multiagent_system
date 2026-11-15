"""Thin LLM wrapper over the OpenAI-compatible Chat Completions API.

Groq, OpenRouter, Google Gemini and local Ollama all expose this API,
so switching provider = changing LLM_PROVIDER / LLM_MODEL in secrets.
"""
from __future__ import annotations

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ddi.config import get_settings

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",  # local dev, no key needed
}

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "gemini": "gemini-flash-latest",
    "ollama": "llama3.1",
}


class LLMNotConfigured(RuntimeError):
    pass


class LLMRateLimited(RuntimeError):
    pass


@retry(retry=retry_if_exception_type(LLMRateLimited),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       stop=stop_after_attempt(4), reraise=True)
def complete(system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 800) -> str:
    """Return the model's text reply."""
    s = get_settings()
    provider = (s.llm_provider or "").lower()
    if provider not in PROVIDERS:
        raise LLMNotConfigured(
            f"LLM_PROVIDER must be one of {list(PROVIDERS)}; got '{s.llm_provider}'.")
    if provider != "ollama" and not s.llm_api_key:
        raise LLMNotConfigured("LLM_API_KEY is empty.")

    model = s.llm_model or DEFAULT_MODELS[provider]
    headers = {"Content-Type": "application/json"}
    if s.llm_api_key:
        headers["Authorization"] = f"Bearer {s.llm_api_key}"

    resp = requests.post(
        f"{PROVIDERS[provider]}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise LLMRateLimited(resp.text[:300])
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()
