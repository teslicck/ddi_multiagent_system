"""Critic agent: check the answer against the original evidence.

Two layers:
  1. deterministic: every citation id exists, every statement has a citation;
  2. LLM (a different model from the synthesizer): each cited claim is really
     supported by the cited fragment, without exaggeration.
The critic reports problems; it does not rewrite the answer (LLM budget).
"""
from __future__ import annotations

from ddi import llm
from ddi.agents.common import (all_evidence, citation_problems, citations,
                               is_not_found_answer, parse_json_object)
from ddi.models import Evidence, SourceReport

SYSTEM = """Ты — агент-критик. Проверь ответ о взаимодействии лекарств по исходным фрагментам.
Для каждого утверждения со ссылкой проверь, что процитированный фрагмент действительно это говорит. Ищи:
- утверждения, которых нет во фрагментах;
- преувеличения: класс препаратов выдан за конкретный препарат, единичный случай — за доказанный факт, «противопоказано» там, где во фрагменте «с осторожностью»;
- неверные ссылки (утверждение есть, но в другом фрагменте).
Мелкие стилистические неточности не считай проблемами.
Верни только JSON без пояснений:
{"verdict": "ok" или "issues", "issues": [{"claim": "<короткая цитата из ответа>", "problem": "<что не так, по-русски>"}]}"""

MAX_FRAGMENT_CHARS = 1200


def _cited_evidence(answer: str, evidence: list[Evidence]) -> list[Evidence]:
    refs = {r.upper() for r in citations(answer)}
    cited = [ev for ev in evidence if ev.ref.upper() in refs]
    return cited or evidence


def critique(answer: str, reports: list[SourceReport]) -> tuple[str, list[str], list[str]]:
    """Return (summary line, issues, errors)."""
    evidence = all_evidence(reports)
    if not evidence:
        return "Проверка не требуется: источники не вернули материалов, ответ это и сообщает.", [], []
    if is_not_found_answer(answer):        # deterministic answer, nothing to verify
        return ("Проверка не требуется: агенты прочитали найденные материалы и не нашли "
                "в них сведений об этой паре; ответ это и сообщает.", [], [])

    issues = citation_problems(answer, evidence)
    errors: list[str] = []

    fragments = "\n\n".join(f"[{ev.ref}] {ev.title}\n{ev.text[:MAX_FRAGMENT_CHARS]}"
                            for ev in _cited_evidence(answer, evidence))
    user = f"Ответ:\n{answer}\n\nИсходные фрагменты:\n\n{fragments}"
    try:
        raw = llm.complete(SYSTEM, user, max_tokens=700, model=llm.critic_model())
        data = parse_json_object(raw)
        if data is None:
            errors.append("Критик вернул ответ не в формате JSON")
        else:
            for item in data.get("issues") or []:
                if isinstance(item, dict) and item.get("problem"):
                    claim = str(item.get("claim", "")).strip()
                    issues.append(f"«{claim}»: {item['problem']}" if claim else str(item["problem"]))
    except Exception as exc:
        errors.append(f"Критик: {exc}")

    if errors and not issues:
        summary = "LLM-проверка не выполнена; формальная проверка ссылок замечаний не нашла."
    elif issues:
        summary = f"Найдено замечаний: {len(issues)}."
    else:
        summary = "Все утверждения подтверждаются процитированными фрагментами."
    return summary, issues, errors
