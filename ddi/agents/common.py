"""Helpers shared by the LLM agents: citation ids, evidence formatting, checks."""
from __future__ import annotations

import json
import re

from ddi.models import Evidence, SourceReport

NOT_FOUND_MARK = "НЕ НАЙДЕНО"       # what a retriever summary says when it has nothing

# Roughly 4 chars per token. Groq's free tier allows 8000 tokens/min per model,
# and the summaries + synthesizer share one model, so each summary gets ~1500
# tokens of evidence.
MAX_EVIDENCE_CHARS = 6000

_REF_TOKEN = re.compile(r"^(DM\d+|PMID:\s?\d+)$", re.I)
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def is_not_found(summary: str) -> bool:
    """True if a retriever summary is the NOT FOUND marker (tolerates '.', '**' etc.)."""
    return re.sub(r"[^\wА-Яа-яЁё ]", "", summary).strip().upper() == NOT_FOUND_MARK


_NOT_FOUND_PREFIX = "В источниках (DailyMed, PubMed) не найдено данных о взаимодействии"


def is_not_found_answer(answer: str) -> bool:
    return answer.startswith(_NOT_FOUND_PREFIX)


def not_found_answer(drug_a: str, drug_b: str) -> str:
    # Never "there is no interaction": absence of evidence is not evidence of absence.
    return (f"{_NOT_FOUND_PREFIX} "
            f"**{drug_a}** и **{drug_b}**. Это не означает, что взаимодействия нет: "
            f"уточните у врача или фармацевта.")


# --------------------------------------------------------------------------- refs

def assign_refs(evidence: list[Evidence]) -> list[Evidence]:
    """Give each piece of evidence a short citation id (in place)."""
    n = 0
    for ev in evidence:
        if ev.source == "pubmed" and ev.meta.get("pmid"):
            ev.ref = f"PMID:{ev.meta['pmid']}"
        else:
            n += 1
            ev.ref = f"DM{n}"
    return evidence


# gpt-oss sometimes cites in OpenAI's own style: 【DM1】 instead of [DM1]
# (seen live, Nov 2026). Map look-alike brackets back before anything parses them.
_BRACKET_LOOKALIKES = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]",
                                     "〔": "[", "〕": "]", "⟦": "[", "⟧": "]"})


def normalize_citations(text: str) -> str:
    return text.translate(_BRACKET_LOOKALIKES)


def _norm(token: str) -> str:
    """'pmid: 123' -> 'PMID:123', 'dm2' -> 'DM2'."""
    return re.sub(r"\s+", "", token).upper()


def citations(text: str) -> list[str]:
    """Citation ids in order of appearance. Accepts [DM1][PMID:1] and [DM1, PMID:1]."""
    out: list[str] = []
    for group in _BRACKETS.findall(normalize_citations(text)):
        for token in re.split(r"[,;]", group):
            if _REF_TOKEN.match(token.strip()):
                ref = _norm(token)
                if ref not in out:
                    out.append(ref)
    return out


def all_evidence(reports: list[SourceReport]) -> list[Evidence]:
    return [ev for r in reports for ev in r.evidence]


# --------------------------------------------------------------------------- formatting

SECTION_LABELS = {
    "boxed_warning": "раздел BOXED WARNING — самое строгое предупреждение FDA",
    "contraindications": "раздел «Противопоказания» (CONTRAINDICATIONS)",
}


def describe(ev: Evidence) -> str:
    m = ev.meta
    if ev.source == "pubmed":
        bits = [b for b in (m.get("year"), ", ".join(m.get("pub_types", [])[:2])) if b]
        return f"{ev.title} ({'; '.join(bits)})" if bits else ev.title
    notes = []
    if m.get("kind") in SECTION_LABELS:
        notes.append(SECTION_LABELS[m["kind"]])
    if m.get("match") == "class":
        notes.append(f"препарат {m.get('about')} здесь не назван; фрагмент найден по классу "
                     f"«{m.get('matched_term')}», к которому он относится")
    return f"{ev.title} ({'; '.join(notes)})" if notes else ev.title


def evidence_block(evidence: list[Evidence], max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    """Numbered fragments for a prompt; stops before the budget is exceeded.

    Evidence arrives best-first from the clients, so the tail is what gets cut.
    """
    parts, used = [], 0
    for ev in evidence:
        part = f"[{ev.ref}] {describe(ev)}\n{ev.text}"
        if parts and used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part) + 2
    return "\n\n".join(parts)


def cited_fragments(answer: str, evidence: list[Evidence], max_chars_each: int) -> str:
    """Original fragments the answer cites (all fragments if it cites none)."""
    refs = {r.upper() for r in citations(answer)}
    cited = [ev for ev in evidence if ev.ref.upper() in refs] or evidence
    return "\n\n".join(f"[{ev.ref}] {describe(ev)}\n{ev.text[:max_chars_each]}" for ev in cited)


# --------------------------------------------------------------------------- checks

_HEADING_ONLY = re.compile(r"^\W*(\*\*[^*]+\*\*|#+ .+)\W*$")


def citation_problems(answer: str, evidence: list[Evidence]) -> list[str]:
    """Deterministic checks: unknown citation ids and statements without a citation."""
    known = {ev.ref.upper() for ev in evidence}
    problems = [f"Ссылка [{ref}] не соответствует ни одному найденному фрагменту."
                for ref in citations(answer) if ref.upper() not in known]
    for line in answer.splitlines():
        text = line.strip()
        if len(text) < 25 or _HEADING_ONLY.match(text):
            continue
        if citations(text) or re.search(r"не найден|нет данных|не нашл", text, re.I):
            continue
        problems.append(f"Утверждение без ссылки на источник: «{_short(text)}»")
    return problems


def _short(text: str, n: int = 160) -> str:
    text = re.sub(r"\*\*", "", text)
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def parse_json_object(text: str) -> dict | None:
    """First {...} object in an LLM reply (models sometimes add prose or ``` fences)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
