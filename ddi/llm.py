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
    # llama-3.3-70b-versatile was retired by Groq (404 model_not_found, Nov 2026).
    # Lighter alternative with ~3-4x fewer output tokens: "qwen/qwen3.8-27b".
    "groq": "openai/gpt-oss-120b",
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

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "gpt-oss" in model:
        # Reasoning model: hidden reasoning counts against max_tokens, and with
        # the default effort a short budget can be spent before any answer.
        payload["reasoning_effort"] = "low"

    resp = requests.post(f"{PROVIDERS[provider]}/chat/completions",
                         headers=headers, json=payload, timeout=60)
    if resp.status_code == 429:
        raise LLMRateLimited(resp.text[:300])
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:300]}")
    choice = resp.json()["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    if not content:
        raise RuntimeError(f"LLM returned no text (finish_reason={choice.get('finish_reason')}); "
                           f"raise max_tokens (now {max_tokens}).")
    return content
