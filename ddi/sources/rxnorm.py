"""Drug name normalization via RxNorm (RxNav REST API).

Docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html

A helper step, not a third data source: it only turns what the user typed into
the active ingredient that DailyMed and PubMed are searched with.
    "Advil" -> ibuprofen, "Zocor" -> simvastatin, "sertraline hydrochloride" ->
    sertraline, typo "warfarn" -> warfarin (approximate match, flagged).

Note: the RxNav *Interaction* API was shut down in January 2024 (see CLAUDE.md);
the RxNorm endpoints used here are separate and still work.

Pure HTTP, no LLM. Never raises for "not found": the caller gets a status and
decides (unrecognized names must not turn into "no interaction found").
"""
from __future__ import annotations

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ddi.config import get_settings
from ddi.models import DrugName, NameStatus
from ddi.sources import rxclass

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class _Transient(Exception):
    pass


@retry(retry=retry_if_exception_type(_Transient),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       stop=stop_after_attempt(3), reraise=True)
def _get(path: str, params: dict | None = None) -> dict:
    try:
        resp = requests.get(f"{BASE_URL}/{path}", params=params,
                            timeout=get_settings().http_timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise _Transient(str(exc)) from exc
    if resp.status_code in (408, 429) or resp.status_code >= 500:
        raise _Transient(f"RxNav HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise RuntimeError(f"RxNav HTTP {resp.status_code} for {path}")
    return resp.json()


def _exact_rxcui(name: str) -> str | None:
    # search=2: normalized string match (case, word order, salt spacing)
    ids = _get("rxcui.json", {"name": name, "search": 2}).get("idGroup", {}).get("rxnormId") or []
    return ids[0] if ids else None


def _approximate(name: str) -> tuple[str, str] | None:
    """Best approximate RxNorm concept (rxcui, name) — catches typos."""
    cands = (_get("approximateTerm.json", {"term": name, "maxEntries": 5})
             .get("approximateGroup", {}).get("candidate") or [])
    best = [c for c in cands if c.get("rank") == "1"]
    named = [c for c in best if c.get("source") == "RXNORM" and c.get("name")]
    pick = (named or best or [None])[0]
    return (pick["rxcui"], pick.get("name") or "") if pick else None


def _name_of(rxcui: str) -> str:
    return _get(f"rxcui/{rxcui}/properties.json").get("properties", {}).get("name", "")


def _ingredients(rxcui: str) -> list[tuple[str, str]]:
    """Active ingredients (TTY=IN) of any concept: brand, salt, ingredient itself.

    Returns [(ingredient rxcui, lowercase name)] without duplicates.
    """
    groups = _get(f"rxcui/{rxcui}/related.json", {"tty": "IN"}) \
        .get("relatedGroup", {}).get("conceptGroup") or []
    out: list[tuple[str, str]] = []
    for g in groups:
        for p in g.get("conceptProperties") or []:
            item = (p.get("rxcui", ""), p["name"].lower())
            if item[1] not in [n for _, n in out]:
                out.append(item)
    return out


def normalize(user_input: str) -> DrugName:
    raw = user_input.strip()
    fallback = raw.lower()
    try:
        rxcui = _exact_rxcui(raw)
        status: NameStatus = "exact"
        matched = ""
        if rxcui is None:
            hit = _approximate(raw)
            if hit is None:
                return DrugName(input=raw, name=fallback, status="not_found")
            rxcui, matched = hit
            status = "approximate"
        ingredients = _ingredients(rxcui)
        if not ingredients:              # obsolete concept (e.g. "Vicodin") or not a drug
            return DrugName(input=raw, name=fallback, status="not_found", rxcui=rxcui)
        # exact match: the input itself is the RxNorm name (saves a ~1 s request)
        matched = matched or (raw if status == "exact" else _name_of(rxcui)) or raw
        names = [n for _, n in ingredients]
        if len(ingredients) > 1:
            return DrugName(input=raw, name=fallback, status="combination", rxcui=rxcui,
                            matched=matched, ingredients=names)
        ing_rxcui, ing_name = ingredients[0]
    except Exception:                    # RxNav down: search with what the user typed
        return DrugName(input=raw, name=fallback, status="unavailable")

    terms, classes, class_error = rxclass.class_terms(ing_rxcui)   # never raises
    return DrugName(input=raw, name=ing_name, status=status, rxcui=rxcui, matched=matched,
                    ingredients=names, ingredient_rxcui=ing_rxcui,
                    classes=classes, class_terms=terms, class_error=class_error or "")
