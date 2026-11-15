"""Critic agent: check the answer against the evidence (unsupported claims, gaps)."""
from __future__ import annotations

from ddi.models import SourceReport


def critique(answer: str, reports: list[SourceReport]) -> str:
    """TODO: LLM checks each claim is backed by a cited fragment."""
    return "(заглушка) Проверка будет реализована после выбора модели."
