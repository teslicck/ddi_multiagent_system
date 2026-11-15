"""DailyMed REST client (FDA labels, section 7 'Drug Interactions').

Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
"""
from __future__ import annotations

from ddi.models import Evidence

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def find_label_setids(drug_name: str, limit: int = 3) -> list[str]:
    """Return SPL set IDs for a drug name. TODO: GET /spls.json?drug_name=..."""
    raise NotImplementedError


def get_interactions_section(setid: str) -> Evidence | None:
    """Fetch the SPL XML and extract the Drug Interactions section. TODO."""
    raise NotImplementedError
