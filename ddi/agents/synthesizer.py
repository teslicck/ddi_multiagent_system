"""Synthesizer agent: merge source reports into one answer with citations."""
from __future__ import annotations

from ddi.models import SourceReport


def synthesize(drug_a: str, drug_b: str, reports: list[SourceReport]) -> str:
    """TODO: prompt the LLM with the evidence; every claim must cite [source]."""
    found = sum(len(r.evidence) for r in reports)
    return f"(заглушка) Найдено фрагментов: {found}. Синтез будет реализован после выбора модели."
