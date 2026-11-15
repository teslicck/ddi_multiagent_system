"""DailyMed REST client (FDA labels: Boxed Warning, Contraindications, Drug Interactions).

Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

Pure HTTP + XML parsing, no LLM. Flow:
    spls.json?drug_name=X  ->  setid  ->  spls/{setid}.xml  ->  Boxed Warning,
    Contraindications, Drug Interactions  ->  passages that mention the other
    drug (or its class), each tagged with the section kind.

Real-world quirks handled here:
  * Many labels are OTC "Drug Facts" without a Drug Interactions section, so the
    search is restricted to human prescription labels (doctype 34391-3).
  * The LOINC code 34073-7 is not always reliable: some old-format labels put it
    on an unrelated section and code the real "Drug Interactions" section as
    unclassified. Sections are therefore matched by code OR by title.
  * Useful drug names often live only in subsection titles
    ("Warfarin-type anticoagulants"), so each passage keeps its section title.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ddi.config import get_settings
from ddi.models import Evidence

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"

RX_DOCTYPE = "34391-3"            # HUMAN PRESCRIPTION DRUG LABEL
DRUG_INTERACTIONS_LOINC = "34073-7"
CONTRAINDICATIONS_LOINC = "34070-3"
BOXED_WARNING_LOINC = "34066-1"
MAX_PASSAGE_CHARS = 1500

# Sections read from a label, most serious first. Contraindications and the
# Boxed Warning hold the most dangerous combinations ("contraindicated with
# lovastatin and simvastatin" in the clarithromycin label), which the Drug
# Interactions section may only cross-reference.
KIND_PRIORITY = {"boxed_warning": 0, "contraindications": 1, "interactions": 2}
# Extra fragments per direction from Boxed Warning / Contraindications, on top
# of `per_direction` Drug Interactions fragments (keeps the LLM token budget small).
CRITICAL_PER_DIRECTION = 1

_NS = "{urn:hl7-org:v3}"
_BLOCK_TAGS = {f"{_NS}paragraph", f"{_NS}item"}
_DI_TITLE = re.compile(r"drug\s+interactions?", re.I)
_LAB_TITLE = re.compile(r"laborator", re.I)
_BOXED_TITLE = re.compile(r"\bboxed\s+warning\b", re.I)
# "4 CONTRAINDICATIONS", "CONTRAINDICATIONS"; not "4.8 Contraindications for Co-administered
# Drugs" at top level (as a subsection it is read together with its parent anyway)
_CONTRA_TITLE = re.compile(r"^\W*(\d+\s+)?contraindications\W*$", re.I)

# Curated synonyms and class names that labels use instead of the drug's own
# name. Since RxClass (ddi/sources/rxclass.py) supplies classes for any drug,
# this list only adds wording RxClass lacks ("strong cyp3a4 inhibitor",
# "anticoagulant" for warfarin, "antiplatelet" for aspirin).
RELATED_TERMS: dict[str, list[str]] = {
    "warfarin": ["coumarin", "anticoagulant", "vitamin k antagonist"],
    "ibuprofen": ["nsaid", "nonsteroidal anti-inflammatory", "non-steroidal anti-inflammatory"],
    "naproxen": ["nsaid", "nonsteroidal anti-inflammatory", "non-steroidal anti-inflammatory"],
    "aspirin": ["salicylate", "nsaid", "antiplatelet"],
    "simvastatin": ["statin", "hmg-coa reductase inhibitor"],
    "atorvastatin": ["statin", "hmg-coa reductase inhibitor"],
    "clarithromycin": ["macrolide", "strong cyp3a4 inhibitor", "strong cyp3a inhibitor"],
    "erythromycin": ["macrolide", "cyp3a4 inhibitor", "cyp3a inhibitor"],
    "ketoconazole": ["azole antifungal", "strong cyp3a4 inhibitor", "strong cyp3a inhibitor"],
    "sertraline": ["ssri", "selective serotonin reuptake inhibitor", "serotonergic"],
    "fluoxetine": ["ssri", "selective serotonin reuptake inhibitor", "serotonergic"],
    "tramadol": ["opioid", "serotonergic"],
    "metformin": ["biguanide"],
    "lisinopril": ["ace inhibitor", "ace-inhibitor", "angiotensin converting enzyme"],
    "digoxin": ["cardiac glycoside", "digitalis"],
}


class DailyMedError(RuntimeError):
    pass


class _Transient(Exception):
    pass


@retry(retry=retry_if_exception_type(_Transient),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       stop=stop_after_attempt(3), reraise=True)
def _get(path: str, params: dict | None = None) -> requests.Response:
    try:
        resp = requests.get(f"{BASE_URL}/{path}", params=params,
                            timeout=get_settings().http_timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise _Transient(str(exc)) from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Transient(f"DailyMed HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise DailyMedError(f"DailyMed HTTP {resp.status_code} for {path}")
    return resp


# --------------------------------------------------------------------------- search

def find_label_setids(drug_name: str, limit: int = 3) -> list[str]:
    """Return set IDs of prescription labels for a drug, single-ingredient first."""
    return [item["setid"] for item in search_labels(drug_name, limit)]


def search_labels(drug_name: str, limit: int = 3) -> list[dict]:
    """Like find_label_setids, but returns the search items ({setid, title, ...})."""
    name = drug_name.strip().lower()
    data = _get("spls.json", {"drug_name": name, "doctype": RX_DOCTYPE,
                              "pagesize": 20}).json().get("data", [])
    # Titles look like "IBUPROFEN TABLET [LABELER]" or "BRAND (A AND B) TABLET [...]".
    # Prefer labels whose title starts with the drug name and has no " AND ".
    def rank(item: dict) -> tuple[int, int]:
        title = item.get("title", "").lower()
        return (0 if title.startswith(name) else 1, 1 if " and " in title else 0)

    return sorted(data, key=rank)[:limit]


# --------------------------------------------------------------------------- parsing

def _text(el: ET.Element) -> str:
    if el.tag == f"{_NS}tr":            # keep table cells apart: "Class | drug, drug"
        cells = [_text(c) for c in el if c.tag in (f"{_NS}td", f"{_NS}th")]
        return " | ".join(c for c in cells if c)
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def section_kind(section: ET.Element) -> str | None:
    """Which of the sections we read this is: 'boxed_warning', 'contraindications',
    'interactions', or None. Matched by LOINC code OR title (codes are not reliable)."""
    code_el = section.find(f"{_NS}code")
    code = code_el.get("code") if code_el is not None else ""
    title_el = section.find(f"{_NS}title")
    title = _text(title_el) if title_el is not None else ""
    if code == BOXED_WARNING_LOINC or _BOXED_TITLE.search(title):
        return "boxed_warning"
    if code == CONTRAINDICATIONS_LOINC or _CONTRA_TITLE.search(title):
        return "contraindications"
    if _LAB_TITLE.search(title):          # "Drug/Laboratory Test Interactions"
        return None
    if code == DRUG_INTERACTIONS_LOINC or _DI_TITLE.search(title):
        return "interactions"
    return None


def _is_interactions_section(section: ET.Element) -> bool:
    return section_kind(section) == "interactions"


def _table_groups(table: ET.Element):
    """Yield table passages. New-format interaction tables are built as groups:
    a one-cell header row ("Strong CYP3A4 inhibitors") followed by rows
    "Clinical Impact: | ...", "Intervention: | ...", "Examples: | ...".
    Such a group becomes one passage so its rows keep their context; rows
    outside any group stay separate passages."""
    group: list[str] = []
    for tr in table.iter(f"{_NS}tr"):
        cells = [c for c in tr if c.tag in (f"{_NS}td", f"{_NS}th")]
        text = _text(tr)
        if not text:
            continue
        if len(cells) == 1:                 # header row starts a new group
            if group:
                yield " / ".join(group)
            group = [text]
        elif group:
            group.append(text)
        else:
            yield text
    if group:
        yield " / ".join(group)


def _blocks(el: ET.Element):
    """Yield text passages (paragraph / list item / table group), skipping
    highlight excerpts and nested sections (those are visited separately)."""
    for child in el:
        if child.tag in (f"{_NS}excerpt", f"{_NS}component", f"{_NS}section"):
            continue
        if child.tag == f"{_NS}table":
            yield from _table_groups(child)
        elif child.tag in _BLOCK_TAGS:
            yield _text(child)
        else:
            yield from _blocks(child)


def _section_passages(section: ET.Element, parent_title: str = "") -> list[tuple[str, str]]:
    """(section title, passage text) for a section and all its subsections."""
    title_el = section.find(f"{_NS}title")
    title = _text(title_el) if title_el is not None else ""
    title = title or parent_title
    out: list[tuple[str, str]] = []
    text_el = section.find(f"{_NS}text")
    if text_el is not None:
        out += [(title, t) for t in _blocks(text_el) if t]
    for sub in section.findall(f"{_NS}component/{_NS}section"):
        out += _section_passages(sub, title)
    return out


def parse_label(xml_bytes: bytes) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (label title, [(kind, section title, passage)]) from SPL XML.

    kind: 'boxed_warning' | 'contraindications' | 'interactions' (see section_kind).
    """
    root = ET.fromstring(xml_bytes)
    doc_title_el = root.find(f"{_NS}title")
    doc_title = _text(doc_title_el) if doc_title_el is not None else "DailyMed label"

    passages: list[tuple[str, str, str]] = []

    def walk(el: ET.Element) -> None:
        for comp in el.findall(f"{_NS}component"):
            for section in comp.findall(f"{_NS}section"):
                kind = section_kind(section)
                if kind:
                    passages.extend((kind, s, t) for s, t in _section_passages(section))
                else:
                    walk(section)
    # structuredBody -> component -> section ...
    for body in root.iter(f"{_NS}structuredBody"):
        walk(body)
    return doc_title, passages


def parse_interactions(xml_bytes: bytes) -> tuple[str, list[tuple[str, str]]]:
    """Return (label title, [(section title, passage)]) of the Drug Interactions section only."""
    doc_title, passages = parse_label(xml_bytes)
    return doc_title, [(s, t) for kind, s, t in passages if kind == "interactions"]


def get_interactions_section(setid: str, title: str | None = None) -> Evidence | None:
    """Fetch a label and return the text of the sections we read, or None.

    Despite the name (kept for callers), this covers Boxed Warning and
    Contraindications as well as Drug Interactions; each passage carries its kind.
    `title` overrides the document title, which for new-format labels is the
    "These highlights do not include..." disclaimer rather than a drug name.
    """
    doc_title, passages = parse_label(_get(f"spls/{setid}.xml").content)
    if not passages:
        return None
    if not title:
        title = doc_title
    return Evidence(
        source="dailymed", title=title, url=LABEL_URL.format(setid=setid),
        text="\n".join(f"[{sec}] {p}" for _, sec, p in passages),
        meta={"setid": setid,
              "passages": [{"kind": k, "section": s, "text": p} for k, s, p in passages]},
    )


# --------------------------------------------------------------------------- filtering

def terms_for(drug: str, extra: list[str] | None = None) -> list[str]:
    """Drug name first, then curated synonyms, then class terms (RxClass)."""
    name = drug.strip().lower()
    terms = [name]
    for t in RELATED_TERMS.get(name, []) + [x.lower() for x in (extra or [])]:
        if t and t not in terms:
            terms.append(t)
    return terms


def filter_passages(passages: list[tuple[str, str]], drug: str,
                    extra_terms: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Keep passages mentioning the drug or a related term (synonym / class).

    Returns (section, text, matched term); exact-name hits come first.
    """
    terms = terms_for(drug, extra_terms)
    patterns = [(t, re.compile(rf"\b{re.escape(t)}", re.I)) for t in terms]
    hits: list[tuple[int, int, str, str, str]] = []
    for sec, text in passages:
        for rank, (term, pat) in enumerate(patterns):
            in_text = bool(pat.search(text))
            if in_text or pat.search(sec):
                # a paragraph that names the drug itself beats one that only sits under a
                # matching heading (live case: "4.5 Lomitapide, Lovastatin, and Simvastatin",
                # whose first paragraph is about lomitapide)
                hits.append((0 if rank == 0 else 1, 0 if in_text else 1, sec, text, term))
                break
    hits.sort(key=lambda h: (h[0], h[1]))   # stable: keeps label order within a rank
    return [(sec, text, term) for _, _, sec, text, term in hits]


_POTENCY = re.compile(r"\b(strong|moderate|weak|potent)\b", re.I)


def _potency_qualified_class_match(hit: tuple[str, str, str], drug: str) -> bool:
    """True for a class-only match inside a group qualified by potency.

    Live case: the simvastatin label's "Strong CYP3A4 inhibitors" group (examples:
    itraconazole, ketoconazole, ...) matched fluconazole via its class "azole";
    the LLM then called fluconazole a strong inhibitor and carried over
    "contraindicated". RxClass has no potency, so such a match would attribute
    another group's recommendation to the drug. Name matches are always kept, and
    so are curated terms that state the potency themselves ("strong cyp3a4
    inhibitor" for clarithromycin in RELATED_TERMS).
    """
    sec, text, term = hit
    if term == drug.strip().lower() or _POTENCY.search(term):
        return False
    return bool(_POTENCY.search(f"{sec} {text}"))


def _clip(text: str, term: str = "", limit: int = MAX_PASSAGE_CHARS) -> str:
    """Shorten a passage to ~limit chars, keeping the part that mentions `term`.

    Big interaction tables can be several thousand chars long with the drug we
    care about near the end, so a plain head cut would drop the actual evidence.
    The group header (text before the first " / ") is kept for context.
    """
    if len(text) <= limit:
        return text
    m = re.search(rf"\b{re.escape(term)}", text, re.I) if term else None
    if m is None or m.end() <= limit - 100:
        return text[:limit].rsplit(" ", 1)[0] + " …"
    header = text.split(" / ", 1)[0][:200]
    budget = limit - len(header) - 6
    start = max(len(header), m.start() - budget // 3)
    window = text[start:start + budget]
    if start > 0:                           # drop the partial first word
        window = window.split(" ", 1)[-1]
    tail = " …" if start + budget < len(text) else ""
    return f"{header} / … {window.rsplit(' ', 1)[0] if tail else window}{tail}"


def find_interactions(drug_a: str, drug_b: str, *, per_direction: int = 4,
                      labels_to_try: int = 3,
                      terms: dict[str, list[str]] | None = None) -> tuple[list[Evidence], list[str]]:
    """Search both directions: A's label about B, and B's label about A.

    `terms` maps a drug name to extra search terms for it (class terms from
    RxClass), used when looking for that drug in the other drug's label.
    Returns (evidence, errors). Errors are human-readable notes, not exceptions,
    so one failed direction does not hide the other.
    """
    terms = terms or {}
    evidence: list[Evidence] = []
    errors: list[str] = []
    for label_drug, other in ((drug_a, drug_b), (drug_b, drug_a)):
        try:
            labels = search_labels(label_drug, limit=labels_to_try)
        except (DailyMedError, _Transient, ValueError) as exc:
            errors.append(f"DailyMed: поиск инструкции для {label_drug} не удался ({exc})")
            continue
        if not labels:
            errors.append(f"DailyMed: инструкция для {label_drug} не найдена")
            continue

        section = None
        for label in labels:               # first label that has the section wins
            try:
                section = get_interactions_section(label["setid"], label.get("title"))
            except (DailyMedError, _Transient, ET.ParseError) as exc:
                errors.append(f"DailyMed: не удалось загрузить {label['setid']} ({exc})")
                continue
            if section:
                break
        if section is None:
            errors.append(f"DailyMed: в инструкции {label_drug} нет разделов о взаимодействиях "
                          f"и противопоказаниях")
            continue

        for kind, sec, text, term in select_hits(section.meta["passages"], other,
                                                 terms.get(other), per_direction):
            evidence.append(Evidence(
                source="dailymed",
                title=f"{section.title} — {sec}",
                url=section.url,
                text=_clip(text, term),
                meta={"setid": section.meta["setid"], "label_drug": label_drug,
                      "about": other, "matched_term": term, "section": sec, "kind": kind,
                      # "class": the label names a class the drug belongs to, not the drug
                      "match": "name" if term == other.strip().lower() else "class"},
            ))
    return evidence, errors


def select_hits(passages: list[dict], drug: str, extra_terms: list[str] | None,
                per_direction: int) -> list[tuple[str, str, str, str]]:
    """Pick the passages about `drug` from one label: (kind, section, text, term).

    Up to `per_direction` from Drug Interactions (as before) plus up to
    CRITICAL_PER_DIRECTION from Boxed Warning / Contraindications. Critical ones
    come first; within each group, name matches before class matches. The same
    potency guard applies everywhere.
    """
    by_kind: dict[str, list[tuple[str, str, str, str]]] = {}
    for kind in KIND_PRIORITY:
        rows = [(p["section"], p["text"]) for p in passages
                if p.get("kind", "interactions") == kind]
        for sec, text, term in filter_passages(rows, drug, extra_terms):
            if not _potency_qualified_class_match((sec, text, term), drug):
                by_kind.setdefault(kind, []).append((kind, sec, text, term))

    name = drug.strip().lower()
    critical = by_kind.get("boxed_warning", []) + by_kind.get("contraindications", [])
    # name match first, then the more serious section (stable sort keeps label order)
    critical.sort(key=lambda h: (h[3] != name, KIND_PRIORITY[h[0]]))
    return critical[:CRITICAL_PER_DIRECTION] + by_kind.get("interactions", [])[:per_direction]
