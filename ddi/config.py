"""Settings: read from Streamlit secrets first, then environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # optional locally
    pass


def _get(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # no secrets file / not running under Streamlit
        pass
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _get("LLM_PROVIDER")
    llm_model: str = _get("LLM_MODEL")
    llm_api_key: str = _get("LLM_API_KEY")
    ncbi_api_key: str = _get("NCBI_API_KEY")
    ncbi_email: str = _get("NCBI_EMAIL")
    http_timeout: float = 20.0
    max_pubmed_results: int = 10


def get_settings() -> Settings:
    return Settings()
