"""Retrieval agents: call a source client, then summarize its evidence with the LLM.

Each agent knows only its own source. The summary cites fragments by id
([DM1], [PMID:123]) so the synthesizer can carry the citations through.
"""
from __future__ import annotations

from ddi import llm
from ddi.agents.common import NOT_FOUND_MARK, assign_refs, evidence_block, normalize_citations
from ddi.models import Evidence, SourceReport
from ddi.sources import dailymed, pubmed

# Fewer, better fragments keep each summary within the Groq token budget.
DAILYMED_PER_DIRECTION = 2
PUBMED_KEEP = 4

_RULES = f"""Правила:
- Используй только текст фрагментов, никаких знаний извне.
- Названия препаратов пиши латиницей, как в запросе (например, tramadol, alprazolam), не транслитерируй.
- После каждого утверждения ставь ссылку на фрагмент в квадратных скобках, как в списке: {{example}}. Несколько ссылок пиши подряд: {{example}}{{example2}}.
- Опиши только то, что есть во фрагментах: эффект или риск, механизм, рекомендацию (противопоказано, избегать, контроль, коррекция дозы).
- Если во фрагментах нет сведений о совместном применении именно этой пары, ответь одной строкой: {NOT_FOUND_MARK}
Формат: 2–5 пунктов списка по-русски, каждый начинается с «- »."""

DAILYMED_SYSTEM = (
    "Ты — агент источника DailyMed (официальные инструкции FDA к препаратам). "
    "Тебе даны фрагменты разделов «Boxed Warning», «Contraindications» и «Drug "
    "Interactions» из инструкций двух препаратов. "
    "Кратко изложи, что они говорят о совместном применении {a} и {b}.\n"
    "Если фрагмент из раздела противопоказаний или BOXED WARNING и в нём назван препарат, "
    "начни с этого пункта и прямо укажи раздел; формулировку («contraindicated», «do not "
    "use») передай точно, не смягчая.\n"
    "Если фрагмент говорит о классе препаратов (например, «NSAIDs», «strong CYP3A4 "
    "inhibitors», «serotonergic drugs»), прямо скажи, что вывод сделан по классу.\n"
    "Если в описании фрагмента сказано, что препарат в нём не назван и фрагмент найден по "
    "классу, проверь список примеров во фрагменте. Если препарата среди примеров нет или "
    "класс уточнён силой действия («strong», «moderate»), не переноси вывод на препарат "
    "автоматически: напиши, что инструкция не называет его прямо и относится ли он к этой "
    "группе, из фрагмента не следует.\n"
    + _RULES.format(example="[DM1]", example2="[DM2]"))

PUBMED_SYSTEM = (
    "Ты — агент источника PubMed (научные статьи). Тебе даны аннотации статей. "
    "Кратко изложи, что они говорят о совместном применении {a} и {b}.\n"
    "Указывай тип работы и год, если они видны (клинический случай, исследование на "
    "добровольцах, обзор). Не выдавай единичный клинический случай за доказанный факт.\n"
    + _RULES.format(example="[PMID:12345]", example2="[PMID:67890]"))


def summarize(system: str, drug_a: str, drug_b: str,
              evidence: list[Evidence]) -> tuple[str, list[str]]:
    """LLM summary of one source's evidence. Returns (summary, errors)."""
    if not evidence:
        return NOT_FOUND_MARK, []          # nothing to read: no LLM call
    user = (f"Препараты: {drug_a} + {drug_b}\n\nФрагменты:\n\n{evidence_block(evidence)}")
    try:
        reply = llm.complete(system.format(a=drug_a, b=drug_b), user, max_tokens=700)
        return normalize_citations(reply), []
    except Exception as exc:               # LLM down / not configured: keep the evidence
        return "", [f"LLM-сводка не получена ({exc})"]


def dailymed_agent(drug_a: str, drug_b: str,
                   terms: dict[str, list[str]] | None = None) -> SourceReport:
    """Find both labels, pull Drug Interactions, keep passages about the other drug.

    `terms`: extra search terms per drug (RxClass classes), see dailymed.find_interactions.
    """
    evidence, errors = dailymed.find_interactions(drug_a, drug_b, terms=terms,
                                                  per_direction=DAILYMED_PER_DIRECTION)
    assign_refs(evidence)
    summary, llm_errors = summarize(DAILYMED_SYSTEM, drug_a, drug_b, evidence)
    return SourceReport(source="dailymed", query=f"{drug_a} + {drug_b}",
                        evidence=evidence, summary=summary, errors=errors + llm_errors)


def pubmed_agent(drug_a: str, drug_b: str) -> SourceReport:
    """Search PubMed for the pair, keep on-topic abstracts, summarize them."""
    evidence, errors = pubmed.find_interactions(drug_a, drug_b, keep=PUBMED_KEEP)
    assign_refs(evidence)
    summary, llm_errors = summarize(PUBMED_SYSTEM, drug_a, drug_b, evidence)
    return SourceReport(source="pubmed", query=pubmed.build_query(drug_a, drug_b),
                        evidence=evidence, summary=summary, errors=errors + llm_errors)
