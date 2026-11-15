"""Synthesizer agent: merge the source summaries into one answer with citations."""
from __future__ import annotations

from ddi import llm
from ddi.agents.common import (NOT_FOUND_MARK, all_evidence, describe, evidence_block,
                               is_not_found, normalize_citations, not_found_answer)
from ddi.models import SourceReport

SOURCE_NAMES = {"dailymed": "DailyMed (инструкции FDA)", "pubmed": "PubMed (научные статьи)"}

SYSTEM = f"""Ты — агент-синтезатор. Тебе даны сводки двух агентов: DailyMed (официальные инструкции FDA) и PubMed (научные статьи). В сводках стоят ссылки вида [DM1] и [PMID:12345].
Составь итоговый ответ по-русски о взаимодействии {{a}} и {{b}}.

Правила:
- Используй только сведения из сводок, ничего не добавляй от себя.
- Названия препаратов пиши латиницей, как в запросе ({{a}}, {{b}}), не транслитерируй.
- Каждое утверждение заканчивай ссылкой (одной или несколькими) ровно в том виде, как в сводках: [DM1], [PMID:12345]. Не придумывай новых ссылок.
- Если сводка источника — «{NOT_FOUND_MARK}», так и напиши про этот источник. Никогда не делай вывод, что взаимодействия нет.
- Если вывод сделан по классу препаратов или по единичным клиническим случаям, скажи об этом.
- Если сводка ссылается на раздел противопоказаний или BOXED WARNING, вынеси это в **Вывод:** и не смягчай формулировку («противопоказано» остаётся «противопоказано»).
- Если источники расходятся, отметь это.
- Не давай медицинских назначений от себя, только пересказ того, что рекомендуют источники.

Формат (Markdown), каждый раздел с новой строки, пропусти раздел, если для него нет данных:
**Вывод:** одна-две фразы: есть ли в источниках данные о взаимодействии и насколько оно серьёзно.
**Что происходит:** эффект и механизм.
**Что рекомендуют источники:** противопоказания, меры предосторожности, контроль.
**Ограничения:** на чём основаны данные."""


def _source_section(report: SourceReport) -> str:
    name = SOURCE_NAMES.get(report.source, report.source)
    if report.summary and not is_not_found(report.summary):
        body = report.summary
    elif report.evidence and not report.summary:   # summary failed: use raw fragments
        body = "Сводка не получена, исходные фрагменты:\n" + evidence_block(report.evidence, 3000)
    else:
        body = NOT_FOUND_MARK
    return f"### {name}\n{body}"


def _fallback(drug_a: str, drug_b: str, reports: list[SourceReport]) -> str:
    """No LLM: list what was found so the user still gets the sources."""
    lines = [f"Не удалось получить итоговую сводку от языковой модели. "
             f"Найденные материалы о {drug_a} + {drug_b}:"]
    lines += [f"- {describe(ev)} [{ev.ref}]" for ev in all_evidence(reports)]
    return "\n".join(lines)


def synthesize(drug_a: str, drug_b: str, reports: list[SourceReport]) -> tuple[str, list[str]]:
    """Return (answer in Markdown with [ref] citations, errors)."""
    if not all_evidence(reports):
        return not_found_answer(drug_a, drug_b), []
    if all(is_not_found(r.summary) for r in reports if r.evidence):
        # the agents read their fragments and found nothing about this pair
        return not_found_answer(drug_a, drug_b), []

    user = f"Препараты: {drug_a} + {drug_b}\n\n" + "\n\n".join(_source_section(r) for r in reports)
    try:
        reply = llm.complete(SYSTEM.format(a=drug_a, b=drug_b), user, max_tokens=900)
        return normalize_citations(reply), []
    except Exception as exc:
        return _fallback(drug_a, drug_b, reports), [f"Синтезатор: {exc}"]
