"""Reviser agent: fix the answer according to the critic's findings.

Works from the ORIGINAL fragments, not from the retriever summaries: live
errors (fluconazole called a "strong" inhibitor, "consider stopping" where the
label says "stop immediately") came from the summaries, so revising against
them would keep the error.

The critic can be wrong too, so the reviser only changes what the fragments
confirm; the orchestrator re-checks the result and keeps whichever version has
fewer findings.
"""
from __future__ import annotations

from ddi import llm
from ddi.agents.common import all_evidence, cited_fragments, normalize_citations
from ddi.models import SourceReport

# Avoid the words used to recognize other agents' prompts in tests.
SYSTEM = """Ты — агент-редактор ответа о взаимодействии лекарств {a} и {b}.
Тебе даны: ответ, список замечаний проверяющего и исходные фрагменты источников со ссылками вида [DM1] и [PMID:12345].

Задача: исправить ответ по замечаниям.
Правила:
- Сверяй каждое замечание с исходными фрагментами. Исправляй только то, что фрагменты подтверждают как ошибку; если замечание фрагментами не подтверждается, оставь это место без изменений.
- Исправление — это уточнение формулировки, смягчение или удаление неподтверждённого утверждения, исправление ссылки. Не добавляй сведений, которых нет во фрагментах.
- Если препарат во фрагменте не назван, а фрагмент о классе препаратов, пиши именно так («инструкция говорит о классе …, сам препарат не назван»).
- Каждое утверждение должно заканчиваться ссылкой ровно в том виде, как во фрагментах. Не придумывай новых ссылок.
- Сохрани структуру и разделы ответа (**Вывод:**, **Что происходит:**, **Что рекомендуют источники:**, **Ограничения:**) и всё, что замечаниями не затронуто.
- Никогда не делай вывод, что взаимодействия нет.
- Названия препаратов пиши латиницей, как в запросе ({a}, {b}); если в ответе название транслитерировано или искажено (например, «тримедол»), замени его.
Верни только исправленный ответ целиком, по-русски, без пояснений о внесённых правках."""

MAX_FRAGMENT_CHARS = 1000      # a bit less than the critic: the reviser shares a model
                               # (and its tokens/min limit) with the summaries and synthesizer


def revise(drug_a: str, drug_b: str, answer: str, issues: list[str],
           reports: list[SourceReport]) -> tuple[str, list[str]]:
    """Return (revised answer, errors). On failure the original answer comes back."""
    if not issues:
        return answer, []
    fragments = cited_fragments(answer, all_evidence(reports), MAX_FRAGMENT_CHARS)
    findings = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, 1))
    user = (f"Ответ:\n{answer}\n\nЗамечания проверяющего:\n{findings}\n\n"
            f"Исходные фрагменты:\n\n{fragments}")
    try:
        reply = llm.complete(SYSTEM.format(a=drug_a, b=drug_b), user, max_tokens=1100)
    except Exception as exc:
        return answer, [f"Редактор: {exc}"]
    return normalize_citations(reply), []
