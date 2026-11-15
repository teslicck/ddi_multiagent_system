"""PubMed client via NCBI E-utilities (esearch + efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Without an API key: ~3 requests/s; with a key: ~10 requests/s.

Pure HTTP + XML parsing, no LLM. Flow:
    esearch (pair + interaction filter)  ->  PMIDs  ->  efetch XML  ->  articles
    ->  keep articles that actually talk about the pair  ->  Evidence.

Relevance sorting tends to surface broad reviews that mention the pair in
passing, and unrelated pairs still get a few hits, so articles are re-ranked
locally: both drugs in the title > both in one sentence about an interaction.
Mere co-mentions and protein-binding / in-silico papers (which use warfarin,
ibuprofen etc. as probes) are dropped.
"""
from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ddi.config import get_settings
from ddi.models import Evidence

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
TOOL_NAME = "ddi-agents"
MAX_ABSTRACT_CHARS = 1500
MIN_SCORE = 2                       # see score_article

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


class PubMedError(RuntimeError):
    pass


class _Transient(Exception):
    pass


# --------------------------------------------------------------------------- HTTP

_rate_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    """Stay under NCBI limits: 3 req/s without a key, 10 req/s with one."""
    global _last_call
    gap = 0.11 if get_settings().ncbi_api_key else 0.34
    with _rate_lock:
        wait = _last_call + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _params(extra: dict) -> dict:
    s = get_settings()
    params = {"db": "pubmed", "tool": TOOL_NAME, **extra}
    if s.ncbi_email:
        params["email"] = s.ncbi_email
    if s.ncbi_api_key:
        params["api_key"] = s.ncbi_api_key
    return params


@retry(retry=retry_if_exception_type(_Transient),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       stop=stop_after_attempt(3), reraise=True)
def _get(endpoint: str, params: dict) -> requests.Response:
    _throttle()
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=_params(params),
                            timeout=get_settings().http_timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise _Transient(str(exc)) from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Transient(f"NCBI HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise PubMedError(f"NCBI HTTP {resp.status_code} for {endpoint}")
    return resp


# --------------------------------------------------------------------------- search

def build_query(drug_a: str, drug_b: str) -> str:
    a, b = (_clean(d) for d in (drug_a, drug_b))
    return f"{a}[tiab] AND {b}[tiab] AND (drug interactions[mh] OR interaction[tiab])"


def _clean(drug: str) -> str:
    # Keep the query syntax ours: drop brackets/quotes/operators users might type.
    return re.sub(r"[\[\]\"()]", " ", drug).strip().lower()


def search(drug_a: str, drug_b: str, max_results: int = 10) -> list[str]:
    """Return PMIDs for '<a> AND <b> AND drug interactions', most relevant first."""
    data = _get("esearch.fcgi", {"term": build_query(drug_a, drug_b), "retmax": max_results,
                                 "retmode": "json", "sort": "relevance"}).json()
    result = data.get("esearchresult", {})
    if "ERROR" in result:
        raise PubMedError(result["ERROR"])
    return list(result.get("idlist", []))


# --------------------------------------------------------------------------- parsing

def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _year(article: ET.Element) -> str:
    pub = article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub is not None:
        year = pub.findtext("Year") or ""
        if not year:                        # e.g. <MedlineDate>1998 Dec-1999 Jan</MedlineDate>
            m = re.search(r"\d{4}", pub.findtext("MedlineDate") or "")
            year = m.group(0) if m else ""
        if year:
            return year
    m = re.search(r"\d{4}", _text(article.find("MedlineCitation/Article/ArticleDate")))
    return m.group(0) if m else ""


def parse_articles(xml_bytes: bytes) -> list[dict]:
    """efetch XML -> [{pmid, title, abstract, year, journal, pub_types, doi}]."""
    root = ET.fromstring(xml_bytes)
    out = []
    for art in root.findall("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        a = cit.find("Article")
        parts = []
        for at in (a.findall("Abstract/AbstractText") if a is not None else []):
            txt = _text(at)
            label = at.get("Label")
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        doi = ""
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
        out.append({
            "pmid": (cit.findtext("PMID") or "").strip(),
            "title": _text(a.find("ArticleTitle")) if a is not None else "",
            "abstract": " ".join(parts),
            "year": _year(art),
            "journal": _text(a.find("Journal/ISOAbbreviation")) or _text(a.find("Journal/Title"))
                       if a is not None else "",
            "pub_types": [_text(p) for p in a.findall("PublicationTypeList/PublicationType")]
                         if a is not None else [],
            "doi": doi,
        })
    return out


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    resp = _get("efetch.fcgi", {"id": ",".join(pmids), "retmode": "xml"})
    articles = parse_articles(resp.content)
    order = {p: i for i, p in enumerate(pmids)}     # keep esearch relevance order
    return sorted(articles, key=lambda x: order.get(x["pmid"], len(order)))


# --------------------------------------------------------------------------- ranking

def _mentions(text: str, drug: str) -> bool:
    return re.search(rf"\b{re.escape(_clean(drug))}", text, re.I) is not None


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


# Words that signal a clinical interaction rather than a mere co-mention
# (e.g. warfarin and ibuprofen used as albumin binding-site markers).
_CLINICAL = re.compile(
    r"\b(interact\w*|co-?administ\w*|concomitant\w*|combin\w*|together|"
    r"bleed\w*|haemorrh\w*|hemorrh\w*|risk\w*|toxicit\w*|adverse|syndrome|"
    r"rhabdomyoly\w*|myopath\w*|contraindicat\w*|inr|exposure|auc|"
    r"increas\w*|decreas\w*|potentiat\w*|inhibit\w*)\b", re.I)


# Biophysics / in-silico papers use classic drugs as probes ("site markers
# warfarin and ibuprofen") and also say "interaction" — not clinical evidence.
_NON_CLINICAL = re.compile(
    r"serum albumin|\bHSA\b|\bBSA\b|binding site|site marker|docking|"
    r"spectroscop\w*|fluorescence quench\w*|molecular dynamics", re.I)


def score_article(article: dict, drug_a: str, drug_b: str) -> int:
    """3 = both drugs in the title; 2 = both in one abstract sentence that also
    talks about an interaction/effect; 1 = both in one sentence (or a protein-
    binding / in-silico paper); 0 = never together."""
    title = article["title"]
    if _mentions(title, drug_a) and _mentions(title, drug_b):
        return 3
    best = 0
    for s in sentences(article["abstract"]):
        if _mentions(s, drug_a) and _mentions(s, drug_b):
            best = 2 if _CLINICAL.search(s) else max(best, 1)
            if best == 2:
                break
    if best == 2 and _NON_CLINICAL.search(f"{title} {article['abstract']}"):
        best = 1
    return best


def excerpt(abstract: str, drug_a: str, drug_b: str, limit: int = MAX_ABSTRACT_CHARS) -> str:
    """Whole abstract if short; otherwise the sentences about the pair, in order.

    Priority: sentences naming both drugs, then the conclusion, then sentences
    naming either drug. Skipped stretches are marked with "…".
    """
    if len(abstract) <= limit:
        return abstract
    sents = sentences(abstract)

    def prio(i: int, s: str) -> int:
        a, b = _mentions(s, drug_a), _mentions(s, drug_b)
        if a and b:
            return 0
        if s.upper().startswith(("CONCLUSION", "CONCLUSIONS")) or i == len(sents) - 1:
            return 1
        return 2 if (a or b) else 3

    chosen: set[int] = set()
    used = 0
    for i, s in sorted(enumerate(sents), key=lambda t: (prio(*t), t[0])):
        if prio(i, s) == 3:
            break
        if used + len(s) + 2 > limit:
            continue
        chosen.add(i)
        used += len(s) + 2
    if not chosen:                          # one giant sentence: plain cut
        return abstract[:limit].rsplit(" ", 1)[0] + " …"

    out, prev = [], -1
    for i in sorted(chosen):
        if i != prev + 1:
            out.append("…")
        out.append(sents[i])
        prev = i
    if prev != len(sents) - 1:
        out.append("…")
    return " ".join(out)


# --------------------------------------------------------------------------- facade

def fetch_abstracts(pmids: list[str]) -> list[Evidence]:
    """Fetch titles/abstracts for PMIDs as Evidence (no filtering)."""
    return [_to_evidence(a, a["abstract"][:MAX_ABSTRACT_CHARS], 0) for a in fetch_articles(pmids)]


def _to_evidence(article: dict, text: str, score: int) -> Evidence:
    meta = {k: article[k] for k in ("pmid", "year", "journal", "pub_types", "doi")}
    meta["score"] = score
    return Evidence(source="pubmed", title=article["title"] or f"PMID {article['pmid']}",
                    url=ARTICLE_URL.format(pmid=article["pmid"]), text=text, meta=meta)


def find_interactions(drug_a: str, drug_b: str, *, max_results: int | None = None,
                      keep: int = 5) -> tuple[list[Evidence], list[str]]:
    """Search, fetch and rank articles about the pair. Returns (evidence, errors)."""
    max_results = max_results or get_settings().max_pubmed_results
    try:
        pmids = search(drug_a, drug_b, max_results)
        articles = fetch_articles(pmids)
    except (PubMedError, _Transient, ET.ParseError, ValueError) as exc:
        return [], [f"PubMed: запрос не удался ({exc})"]

    ranked = []
    for art in articles:
        score = score_article(art, drug_a, drug_b)
        if score < MIN_SCORE:               # mere co-mention / binding-assay papers
            continue
        if not art["abstract"] and score < 3:   # no abstract: only keep on-topic titles
            continue
        ranked.append((score, art))
    ranked.sort(key=lambda t: -t[0])             # stable: relevance order within a score

    evidence = [_to_evidence(art, excerpt(art["abstract"], drug_a, drug_b) or art["title"], score)
                for score, art in ranked[:keep]]
    return evidence, []
