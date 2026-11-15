"""Boxed Warning and Contraindications sections — offline, on a trimmed real label."""
from __future__ import annotations

from pathlib import Path

import pytest

from ddi.agents import common
from ddi.models import Evidence
from ddi.sources import dailymed

LABEL = (Path(__file__).parent / "fixtures" / "dailymed" / "label_sections.xml").read_bytes()


def test_parse_label_tags_sections_by_kind():
    _, passages = dailymed.parse_label(LABEL)
    kinds = {}
    for kind, sec, text in passages:
        kinds.setdefault(kind, []).append(sec)
    assert set(kinds) == {"boxed_warning", "contraindications", "interactions"}
    assert kinds["boxed_warning"] == [""]                      # untitled, found by LOINC code
    assert kinds["contraindications"] == ["4.1 Hypersensitivity",
                                          "4.5 Lomitapide, Lovastatin, and Simvastatin",
                                          "4.5 Lomitapide, Lovastatin, and Simvastatin"]
    assert not any("must not be read" in t for _, _, t in passages)   # 4.8 is not section 4


def test_parse_interactions_still_returns_only_drug_interactions():
    _, passages = dailymed.parse_interactions(LABEL)
    assert passages and all(sec == "7 DRUG INTERACTIONS" for sec, _ in passages)


@pytest.mark.parametrize("title, expected", [
    ("4 CONTRAINDICATIONS", True), ("CONTRAINDICATIONS", True), ("Contraindications", True),
    ("4.8 Contraindications for Co-administered Drugs", False), ("Drug Interactions", False),
])
def test_contraindications_title(title, expected):
    assert bool(dailymed._CONTRA_TITLE.search(title)) is expected


def label_passages():
    _, passages = dailymed.parse_label(LABEL)
    return [{"kind": k, "section": s, "text": t} for k, s, t in passages]


def test_select_hits_one_critical_plus_drug_interactions():
    hits = dailymed.select_hits(label_passages(), "simvastatin", ["statin"], per_direction=2)
    assert [k for k, *_ in hits] == ["contraindications", "interactions", "interactions"]
    kind, sec, text, term = hits[0]
    # the lomitapide paragraph matches only via the heading; the one naming simvastatin wins
    assert sec.startswith("4.5") and term == "simvastatin" and "Do not use" in text
    assert not any("beyond the per-direction limit" in t for _, _, t, _ in hits)


def test_boxed_warning_wins_among_name_matches():
    passages = [{"kind": "contraindications", "section": "4", "text": "Do not use with simvastatin."},
                {"kind": "boxed_warning", "section": "", "text": "Fatal myopathy with simvastatin."}]
    hits = dailymed.select_hits(passages, "simvastatin", None, per_direction=2)
    assert [k for k, *_ in hits] == ["boxed_warning"]


def test_name_match_beats_class_match_in_critical_sections():
    passages = [{"kind": "boxed_warning", "section": "", "text": "Statins may cause myopathy."},
                {"kind": "contraindications", "section": "4", "text": "Do not use with simvastatin."}]
    hits = dailymed.select_hits(passages, "simvastatin", ["statin"], per_direction=2)
    assert [(k, t) for k, _, _, t in hits] == [("contraindications", "simvastatin")]


def test_potency_guard_applies_to_critical_sections():
    passages = [{"kind": "contraindications", "section": "4",
                 "text": "Concomitant use of strong CYP3A4 inhibitors (select azole anti-fungals)."}]
    assert dailymed.select_hits(passages, "fluconazole", ["azole"], 2) == []          # class only
    kept = dailymed.select_hits(passages, "clarithromycin", ["strong cyp3a4 inhibitor"], 2)
    assert kept and kept[0][3] == "strong cyp3a4 inhibitor"                           # curated


def test_find_interactions_marks_kind_and_llm_sees_section(monkeypatch):
    monkeypatch.setattr(dailymed, "search_labels",
                        lambda drug, limit: [{"setid": drug, "title": drug.upper()}])
    monkeypatch.setattr(dailymed, "_get", lambda path, params=None: type(
        "R", (), {"content": LABEL if "clarithromycin" in path else b"<document xmlns='urn:hl7-org:v3'/>"})())
    ev, errors = dailymed.find_interactions("simvastatin", "clarithromycin", per_direction=2)
    from_clari = [e for e in ev if e.meta["label_drug"] == "clarithromycin"]
    assert [e.meta["kind"] for e in from_clari] == ["contraindications", "interactions", "interactions"]
    assert "Do not use clarithromycin" in from_clari[0].text
    assert "раздел «Противопоказания»" in common.describe(from_clari[0])
    assert any("simvastatin нет разделов" in e for e in errors)   # empty simvastatin label


def test_describe_boxed_and_class_notes_combined():
    ev = Evidence(source="dailymed", title="T", url="u", text="x",
                  meta={"kind": "boxed_warning", "match": "class", "about": "meloxicam",
                        "matched_term": "nsaid"})
    d = common.describe(ev)
    assert "BOXED WARNING" in d and "найден по классу «nsaid»" in d
