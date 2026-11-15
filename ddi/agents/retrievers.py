"""Retrieval agents: call a source client, then (optionally) summarize with the LLM."""
from __future__ import annotations

from ddi.models import SourceReport


def dailymed_agent(drug_a: str, drug_b: str) -> SourceReport:
    """Find both labels, pull section 7, keep passages mentioning the other drug. TODO."""
    return SourceReport(source="dailymed", query=f"{drug_a} + {drug_b}",
                        errors=["DailyMed agent not implemented yet"])


def pubmed_agent(drug_a: str, drug_b: str) -> SourceReport:
    """Search PubMed for the pair and summarize the abstracts. TODO."""
    return SourceReport(source="pubmed", query=f"{drug_a} AND {drug_b}",
                        errors=["PubMed agent not implemented yet"])
