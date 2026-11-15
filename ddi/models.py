"""Shared data structures passed between agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NameStatus = Literal["exact", "approximate", "combination", "not_found", "unavailable"]


class DrugName(BaseModel):
    """A user-entered drug name normalized via RxNorm (ddi/sources/rxnorm.py)."""

    input: str
    name: str                    # what the sources are searched with (lowercase ingredient)
    status: NameStatus
    rxcui: str = ""
    matched: str = ""            # the RxNorm concept the input matched ("Advil", "warfarin")
    ingredients: list[str] = Field(default_factory=list)
    ingredient_rxcui: str = ""
    classes: list[str] = Field(default_factory=list)      # FDA classes (EPC), for display
    class_terms: list[str] = Field(default_factory=list)  # label search terms from RxClass
    class_error: str = ""        # RxClass failed: search falls back to name + curated terms

    @property
    def changed(self) -> bool:
        return self.name != self.input.strip().lower()


class Evidence(BaseModel):
    """One piece of retrieved evidence with its origin."""

    source: str                 # "dailymed" | "pubmed"
    ref: str = ""               # citation id used by the LLM agents: "DM1", "PMID:123"
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
    issues: list[str] = Field(default_factory=list)   # critic findings on `answer`
    errors: list[str] = Field(default_factory=list)   # synthesizer / critic / reviser failures
    # Revision round: the synthesizer's first answer and what the critic found in it.
    draft: str = ""
    draft_issues: list[str] = Field(default_factory=list)
    revised: bool = False        # True: `answer` is the reviser's version
    revision_note: str = ""      # what happened in the revision round, for the UI
    reports: list[SourceReport] = Field(default_factory=list)
    drugs: list[DrugName] = Field(default_factory=list)   # normalized inputs, [a, b]
    stopped: bool = False        # True: input rejected before searching (unknown / combo / same)
