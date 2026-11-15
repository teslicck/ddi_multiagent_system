"""Orchestrator: runs retrievers in parallel, then synthesizer, then critic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ddi.agents.critic import critique
from ddi.agents.retrievers import dailymed_agent, pubmed_agent
from ddi.agents.synthesizer import synthesize
from ddi.models import FinalReport


def run(drug_a: str, drug_b: str) -> FinalReport:
    drug_a, drug_b = drug_a.strip(), drug_b.strip()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dailymed_agent, drug_a, drug_b),
                   pool.submit(pubmed_agent, drug_a, drug_b)]
        reports = [f.result() for f in futures]

    answer = synthesize(drug_a, drug_b, reports)
    review = critique(answer, reports)
    return FinalReport(drug_a=drug_a, drug_b=drug_b, answer=answer,
                       critique=review, reports=reports)
