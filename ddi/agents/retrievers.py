"""Retrieval agents: call a source client, then (optionally) summarize with the LLM."""
from __future__ import annotations

from ddi.models import SourceReport
from ddi.sources import dailymed, pubmed


def dailymed_agent(drug_a: str, drug_b: str) -> SourceReport:
    """Find both labels, pull Drug Interactions, keep passages mentioning the other drug.

    TODO: short LLM summary of the evidence (step 3 in CLAUDE.md).
    """
    evidence, errors = dailymed.find_interactions(drug_a, drug_b)
    return SourceReport(source="dailymed", query=f"{drug_a} + {drug_b}",
                        evidence=evidence, errors=errors)


def pubmed_agent(drug_a: str, drug_b: str) -> SourceReport:
    """Search PubMed for the pair, keep on-topic abstracts.

    TODO: short LLM summary of the evidence (step 3 in CLAUDE.md).
    """
    evidence, errors = pubmed.find_interactions(drug_a, drug_b)
    return SourceReport(source="pubmed", query=pubmed.build_query(drug_a, drug_b),
                        evidence=evidence, errors=errors)
