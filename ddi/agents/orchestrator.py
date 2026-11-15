"""Orchestrator: normalize names -> retrievers (parallel) -> synthesizer -> critic
-> reviser -> critic again.

Step 0 (no LLM): RxNorm turns what the user typed into active ingredients
("Advil" -> ibuprofen). Unrecognized names and combination products stop the
pipeline with an explicit message: an unknown name must never come back as
"no interaction data found".

Revision round: if the critic found problems, the reviser fixes them against
the original fragments and the critic checks the new version. The version with
fewer findings is shown (the critic can be wrong, so a "fix" is not trusted
blindly); the draft and its findings stay available for the UI.

LLM budget per user query: 2 summaries (parallel) + synthesizer + critic, plus
reviser + second critic pass when there are findings = 6 calls at most; fewer
when a source returns nothing (no evidence -> no LLM call).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ddi.agents.common import is_not_found_answer
from ddi.agents.critic import critique
from ddi.agents.retrievers import dailymed_agent, pubmed_agent
from ddi.agents.reviser import revise
from ddi.agents.synthesizer import synthesize
from ddi.models import DrugName, FinalReport
from ddi.sources import rxnorm


def input_problem(a: DrugName, b: DrugName) -> str | None:
    """Message explaining why the pair cannot be searched, or None."""
    problems = []
    for d in (a, b):
        if d.status == "not_found":
            problems.append(
                f"Препарат «{d.input}» не распознан в RxNorm. Проверьте написание: нужно "
                f"международное (INN) или торговое название на английском, например "
                f"ibuprofen или Advil.")
        elif d.status == "combination":
            problems.append(
                f"«{d.input}» — комбинированный препарат ({' + '.join(d.ingredients)}). "
                f"Укажите одно действующее вещество и проверьте пары по очереди.")
    if not problems and a.name == b.name:
        problems.append(f"«{a.input}» и «{b.input}» — один и тот же препарат ({a.name}).")
    return "\n\n".join(problems) or None


def revision_round(drug_a: str, drug_b: str, draft: str, draft_review: str,
                   draft_issues: list[str], reports) -> dict:
    """Fix the draft by the critic's findings, re-check, keep the better version.

    Returns the FinalReport fields for answer / critique / issues / revision.
    """
    keep_draft = {"answer": draft, "critique": draft_review, "issues": draft_issues,
                  "draft": "", "draft_issues": [], "revised": False, "revision_note": ""}
    if not draft_issues or is_not_found_answer(draft):
        return {**keep_draft, "errors": []}

    revised, errors = revise(drug_a, drug_b, draft, draft_issues, reports)
    if errors or revised.strip() == draft.strip():
        return {**keep_draft, "errors": errors,
                "revision_note": "Исправление по замечаниям не получено; показан исходный ответ."}

    review, issues, critic_errors = critique(revised, reports)
    base = {"draft": draft, "draft_issues": draft_issues, "errors": critic_errors}
    if critic_errors:
        # revised version could not be verified: show it, but say so
        return {**base, "answer": revised, "critique": review, "issues": issues,
                "revised": True,
                "revision_note": "Ответ исправлен по замечаниям, но повторная проверка не выполнена."}
    if len(issues) > len(draft_issues):
        return {**base, "answer": draft, "critique": draft_review, "issues": draft_issues,
                "revised": False,
                "revision_note": (f"Исправленный вариант отклонён: повторная проверка нашла больше "
                                  f"замечаний ({len(issues)} против {len(draft_issues)}).")}
    return {**base, "answer": revised, "critique": review, "issues": issues, "revised": True,
            "revision_note": (f"Ответ исправлен по замечаниям критика: было {len(draft_issues)}, "
                              f"после повторной проверки {len(issues)}.")}


def run(drug_a: str, drug_b: str) -> FinalReport:
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(rxnorm.normalize, (drug_a, drug_b))

    problem = input_problem(a, b)
    if problem:
        return FinalReport(drug_a=a.input, drug_b=b.input, answer=problem,
                           drugs=[a, b], stopped=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        class_terms = {a.name: a.class_terms, b.name: b.class_terms}
        futures = [pool.submit(dailymed_agent, a.name, b.name, class_terms),
                   pool.submit(pubmed_agent, a.name, b.name)]
        reports = [f.result() for f in futures]

    draft, synth_errors = synthesize(a.name, b.name, reports)
    review, issues, critic_errors = critique(draft, reports)
    result = revision_round(a.name, b.name, draft, review, issues, reports)
    errors = synth_errors + critic_errors + result.pop("errors")
    return FinalReport(drug_a=a.name, drug_b=b.name, errors=errors, reports=reports,
                       drugs=[a, b], **result)
