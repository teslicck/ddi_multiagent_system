"""PubMed client via NCBI E-utilities (esearch + efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Without an API key: ~3 requests/s; with a key: ~10 requests/s.
"""
from __future__ import annotations

from ddi.models import Evidence

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search(drug_a: str, drug_b: str, max_results: int = 10) -> list[str]:
    """Return PMIDs for '<a> AND <b> AND drug interactions'. TODO: esearch.fcgi."""
    raise NotImplementedError


def fetch_abstracts(pmids: list[str]) -> list[Evidence]:
    """Fetch titles/abstracts for PMIDs. TODO: efetch.fcgi (XML)."""
    raise NotImplementedError
