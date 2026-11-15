"""Drug classes via RxClass (RxNav) -> search terms for DailyMed labels.

Docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html

Labels often name a class instead of the drug: the lisinopril label warns about
"NSAIDs", not meloxicam; the tramadol label about "SSRIs", not escitalopram.
A hand-written class list covered ~15 drugs; RxClass covers any ingredient.

Which classes (checked live on 12 drugs, Nov 2026):
  * FDASPL EPC / MoA — the classes FDA itself assigns in labels, so their wording
    matches label text ("Nonsteroidal Anti-inflammatory Drug", "Cytochrome P450
    3A4 Inhibitors", "Serotonin and Norepinephrine Reuptake Inhibitor").
  * Any FDA class mentioning serotonin -> also "serotonergic", the word labels use
    in serotonin-syndrome warnings (SSRIs, SNRIs, triptans).
  * Not ATC / VA: they list combination-product classes (ibuprofen gets
    "Opioid analgesics" from ibuprofen+oxycodone), which would add false matches.
  * Only relaSource=FDASPL is requested: 2-5 items instead of 70-130, and the
    full list sometimes took 8-24 s to arrive.

Helper step like RxNorm, not a data source. Pure HTTP, no LLM.
"""
from __future__ import annotations

import re

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxclass"

# How labels abbreviate or rephrase a class. Keyed by the singular, lowercase
# class phrase produced by _phrase(); values are added as extra terms.
CLASS_ALIASES: dict[str, list[str]] = {
    "nonsteroidal anti-inflammatory drug": ["nsaid", "nonsteroidal anti-inflammatory",
                                            "non-steroidal anti-inflammatory"],
    "serotonin reuptake inhibitor": ["ssri", "selective serotonin reuptake inhibitor"],
    "serotonin and norepinephrine reuptake inhibitor": ["snri"],
    "angiotensin converting enzyme inhibitor": ["ace inhibitor", "ace-inhibitor"],
    "angiotensin-converting enzyme inhibitor": ["ace inhibitor", "ace-inhibitor"],
    "angiotensin 2 receptor blocker": ["angiotensin receptor blocker", "arb"],
    "hmg-coa reductase inhibitor": ["statin"],
    "opioid agonist": ["opioid"],
    "macrolide antimicrobial": ["macrolide"],
    "vitamin k antagonist": ["anticoagulant", "coumarin"],
    "monoamine oxidase inhibitor": ["maoi", "mao inhibitor"],
    "azole antifungal": ["azole"],
    "beta adrenergic blocker": ["beta-blocker", "beta blocker"],
    "calcium channel blocker": ["calcium channel blocker"],
    "p-glycoprotein inhibitor": ["p-gp inhibitor"],
    "p-glycoprotein inducer": ["p-gp inducer"],
}

# Phrases too generic to search label text with (they match unrelated passages).
TOO_BROAD = {"cyclooxygenase inhibitor", "full opioid agonist", "serotonin uptake inhibitor",
             "norepinephrine uptake inhibitor", "decreased coagulation factor activity"}

_CYP = re.compile(r"^cytochrome p450 (\w+) (inhibitor|inducer)$")
TIMEOUT_S = 10.0


def _phrase(class_name: str) -> str:
    """'Cytochrome P450 3A4 Inhibitors' -> 'cytochrome p450 3a4 inhibitor'.

    Singular form: search terms are matched as word prefixes, so the singular
    also hits the plural in label text.
    """
    p = re.sub(r"\s*\[.*?\]$", "", class_name).strip().lower()
    words = p.split()
    if words and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def terms_from_classes(classes: list[dict]) -> list[str]:
    """RxClass items ({className, classType, relaSource, rela}) -> search terms."""
    terms: list[str] = []

    def add(t: str) -> None:
        if t and t not in terms and t not in TOO_BROAD:
            terms.append(t)

    for c in classes:
        source, ctype, name = c.get("relaSource"), c.get("classType"), c.get("className", "")
        if source != "FDASPL" or ctype not in ("EPC", "MOA"):
            continue
        phrase = _phrase(name)
        add(phrase)
        m = _CYP.match(phrase)
        if m:                                   # labels write "CYP3A4 inhibitors"
            add(f"cyp{m.group(1)} {m.group(2)}")
        for alias in CLASS_ALIASES.get(phrase, []):
            add(alias)
        if "serotonin" in phrase:
            add("serotonergic")
    return terms


class _Transient(Exception):
    pass


@retry(retry=retry_if_exception_type(_Transient),
       wait=wait_exponential(multiplier=1, min=1, max=4),
       stop=stop_after_attempt(2), reraise=True)
def get_classes(rxcui: str) -> list[dict]:
    """FDA (FDASPL) class memberships of an ingredient: EPC, MoA, PE, chemical structure."""
    try:
        resp = requests.get(f"{BASE_URL}/class/byRxcui.json",
                            params={"rxcui": rxcui, "relaSource": "FDASPL"}, timeout=TIMEOUT_S)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise _Transient(str(exc)) from exc
    if resp.status_code in (408, 429) or resp.status_code >= 500:
        raise _Transient(f"RxClass HTTP {resp.status_code}")
    resp.raise_for_status()
    items = (resp.json().get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", []) or []
    return [{**it.get("rxclassMinConceptItem", {}),
             "relaSource": it.get("relaSource"), "rela": it.get("rela")} for it in items]


def class_terms(rxcui: str) -> tuple[list[str], list[str], str | None]:
    """(search terms, human-readable FDA class names, error) for an ingredient rxcui.

    Never raises. On failure returns ([], [], "<why>"): the caller still has the
    drug name and the curated terms, but recall drops, so the error is shown.
    """
    if not rxcui:
        return [], [], None
    try:
        classes = get_classes(rxcui)
    except Exception as exc:
        return [], [], f"RxClass недоступен ({type(exc).__name__})"
    names = []
    for c in classes:
        if c.get("relaSource") == "FDASPL" and c.get("classType") == "EPC":
            n = re.sub(r"\s*\[.*?\]$", "", c.get("className", ""))
            if n and n not in names:
                names.append(n)
    return terms_from_classes(classes), names, None
