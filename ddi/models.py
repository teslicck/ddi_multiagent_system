"""Shared data structures passed between agents."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One piece of retrieved evidence with its origin."""

    source: str                 # "dailymed" | "pubmed"
    title: str
    url: str
    text: str                   # excerpt the agents reason over
    meta: dict = Field(default_factory=dict)  # setid, pmid, year, etc.


class SourceReport(BaseModel):
    """Output of a retrieval agent."""

    source: str
    query: str
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str = ""
    errors: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Output of the orchestrator shown in the UI."""

    drug_a: str
    drug_b: str
    answer: str
    critique: str = ""
    reports: list[SourceReport] = Field(default_factory=list)
