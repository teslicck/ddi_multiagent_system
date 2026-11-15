"""Orchestrator: normalizes names, runs retrievers in parallel, then synthesizer and critic.

Step 0 (no LLM): RxNorm turns what the user typed into active ingredients
("Advil" -> ibuprofen). Unrecognized names and combination products stop the
pipeline with an explicit message: an unknown name must never come back as
"no interaction data found".

LLM budget per user query: 2 summaries (parallel) + synthesizer + critic = 4 calls
at most; fewer when a source returns nothing (no evidence -> no LLM call).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ddi.agents.critic import critique
from ddi.agents.retrievers import dailymed_agent, pubmed_agent
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


def run(drug_a: str, drug_b: str) -> FinalReport:
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(rxnorm.normalize, (drug_a, drug_b))

    problem = input_problem(a, b)
    if problem:
        return FinalReport(drug_a=a.input, drug_b=b.input, answer=problem,
                           drugs=[a, b], stopped=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dailymed_agent, a.name, b.name),
                   pool.submit(pubmed_agent, a.name, b.name)]
        reports = [f.result() for f in futures]

    answer, synth_errors = synthesize(a.name, b.name, reports)
    review, issues, critic_errors = critique(answer, reports)
    return FinalReport(drug_a=a.name, drug_b=b.name, answer=answer, critique=review,
                       issues=issues, errors=synth_errors + critic_errors, reports=reports,
                       drugs=[a, b])
