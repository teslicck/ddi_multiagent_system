"""RxClass class terms on saved real RxClass replies — no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ddi.agents import common
from ddi.models import Evidence
from ddi.sources import dailymed, rxclass

FIXTURES = Path(__file__).parent / "fixtures" / "rxclass"


def classes_from_fixture(name: str) -> list[dict]:
    data = json.loads((FIXTURES / f"fdaspl_{name}.json").read_text(encoding="utf-8"))
    return [{**it["rxclassMinConceptItem"], "relaSource": it["relaSource"], "rela": it["rela"]}
            for it in data["rxclassDrugInfoList"]["rxclassDrugInfo"]]


def cls(name, ctype="EPC", source="FDASPL"):
    return {"className": name, "classType": ctype, "relaSource": source}


# --------------------------------------------------------------------------- terms

def test_phrase_singular_lowercase_no_suffix():
    assert rxclass._phrase("Cytochrome P450 3A4 Inhibitors") == "cytochrome p450 3a4 inhibitor"
    assert rxclass._phrase("Nonsteroidal Anti-inflammatory Drug [EPC]") == \
           "nonsteroidal anti-inflammatory drug"
    assert rxclass._phrase("Azoles") == "azole"
    assert rxclass._phrase("Glass") == "glass"                 # no "-ss" -> "-s"


def test_fluconazole_real_reply_gives_cyp_terms():
    terms = rxclass.terms_from_classes(classes_from_fixture("fluconazole"))
    assert "azole antifungal" in terms and "azole" in terms
    for cyp in ("cyp3a4 inhibitor", "cyp2c9 inhibitor", "cyp2c19 inhibitor"):
        assert cyp in terms                                   # labels say "CYP3A4 inhibitors"


def test_tramadol_and_ibuprofen_real_replies():
    tram = rxclass.terms_from_classes(classes_from_fixture("tramadol"))
    assert "opioid" in tram and "opioid agonist" in tram
    assert "full opioid agonist" not in tram                 # too broad, filtered
    ibu = rxclass.terms_from_classes(classes_from_fixture("ibuprofen"))
    assert {"nsaid", "nonsteroidal anti-inflammatory"} <= set(ibu)
    assert "cyclooxygenase inhibitor" not in ibu             # too broad, filtered


def test_serotonin_classes_add_serotonergic_and_aliases():
    terms = rxclass.terms_from_classes([cls("Serotonin Reuptake Inhibitor"),
                                        cls("Serotonin Uptake Inhibitors", "MOA")])
    assert terms[:1] == ["serotonin reuptake inhibitor"]
    assert {"ssri", "selective serotonin reuptake inhibitor", "serotonergic"} <= set(terms)
    assert "serotonin uptake inhibitor" not in terms
    assert terms.count("serotonergic") == 1


def test_non_fda_sources_are_ignored():
    # ATC/VA list combination-product classes: ibuprofen would become an "opioid"
    terms = rxclass.terms_from_classes([cls("OPIOID ANALGESICS", "VA", "VA"),
                                        cls("Opioids in combination", "ATC1-4", "ATC"),
                                        cls("Increased Serotonin Activity", "PE", "MEDRT")])
    assert terms == []


# --------------------------------------------------------------------------- HTTP

class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self._data, self.status_code = data, status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_class_terms_requests_fdaspl_only(monkeypatch):
    seen = {}
    data = json.loads((FIXTURES / "fdaspl_tramadol.json").read_text(encoding="utf-8"))

    def fake_get(url, params=None, timeout=None):
        seen.update(params)
        return FakeResponse(data)

    monkeypatch.setattr(rxclass.requests, "get", fake_get)
    terms, names, error = rxclass.class_terms("10689")
    assert seen == {"rxcui": "10689", "relaSource": "FDASPL"}
    assert names == ["Opioid Agonist"] and "opioid" in terms and error is None


def test_class_terms_failure_is_reported_not_raised(monkeypatch):
    calls = []

    def boom(url, params=None, timeout=None):
        calls.append(1)
        raise requests.Timeout("slow")

    monkeypatch.setattr(rxclass.requests, "get", boom)
    monkeypatch.setattr(rxclass.get_classes.retry, "sleep", lambda _: None)
    terms, names, error = rxclass.class_terms("4450")
    assert (terms, names) == ([], []) and "RxClass недоступен" in error
    assert len(calls) == 2                                    # one retry


def test_no_rxcui_no_request(monkeypatch):
    monkeypatch.setattr(rxclass.requests, "get", lambda *a, **k: pytest.fail("no request"))
    assert rxclass.class_terms("") == ([], [], None)


# --------------------------------------------------------------------------- DailyMed

LISINOPRIL_PASSAGES = [
    ("7.1 Diuretics", "Initiation of lisinopril in patients on diuretics may cause hypotension."),
    ("7.3 Non-Steroidal Anti-Inflammatory Agents Including COX-2 Inhibitors",
     "Coadministration of NSAIDs with ACE inhibitors may result in deterioration of renal function."),
]


def test_class_terms_find_passage_that_never_names_the_drug():
    # live case: the lisinopril label warns about "NSAIDs", never "meloxicam"
    assert dailymed.filter_passages(LISINOPRIL_PASSAGES, "meloxicam") == []
    meloxicam_terms = rxclass.terms_from_classes([cls("Nonsteroidal Anti-inflammatory Drug")])
    hits = dailymed.filter_passages(LISINOPRIL_PASSAGES, "meloxicam", meloxicam_terms)
    assert [(s[:3], term) for s, _, term in hits] == [("7.3", "nsaid")]   # "NSAIDs" in text


def test_name_match_outranks_class_match():
    passages = [("A", "NSAIDs increase bleeding."), ("B", "Meloxicam raised INR in one case.")]
    hits = dailymed.filter_passages(passages, "meloxicam", ["nsaid"])
    assert [s for s, _, _ in hits] == ["B", "A"]


def test_potency_qualified_class_match_is_dropped_name_match_kept(monkeypatch):
    # live case: simvastatin label, "Strong CYP3A4 inhibitors" group, fluconazole not listed
    strong = ("7.1 Drug Interactions", "Strong CYP3A4 inhibitors / Clinical Impact: | Increased "
              "exposure. / Examples: | Select azole antifungals (itraconazole, ketoconazole)")
    named = ("7.1 Drug Interactions", "Strong CYP3A4 inhibitors / Examples: | ketoconazole")
    general = ("7.2", "Azole antifungals may increase simvastatin exposure.")
    monkeypatch.setattr(dailymed, "search_labels",
                        lambda drug, limit: [{"setid": drug, "title": drug.upper()}])
    monkeypatch.setattr(dailymed, "get_interactions_section", lambda setid, title: Evidence(
        source="dailymed", title=title, url="u", text="", meta={"setid": setid, "passages": [
            {"section": s, "text": t} for s, t in ([strong, named, general]
                                                   if setid == "simvastatin" else [])]}))

    ev, _ = dailymed.find_interactions("fluconazole", "simvastatin",
                                       terms={"fluconazole": ["azole"]})
    assert [e.meta["section"] for e in ev] == ["7.2"]          # strong group dropped

    ev, _ = dailymed.find_interactions("ketoconazole", "simvastatin",
                                       terms={"ketoconazole": ["azole"]})
    assert ev[0].meta["match"] == "name" and "Strong" in ev[0].text   # named -> kept

    # curated term that itself states potency is trusted
    assert not dailymed._potency_qualified_class_match(
        ("7.1", strong[1], "strong cyp3a4 inhibitor"), "clarithromycin")


def test_find_interactions_marks_class_matches(monkeypatch):
    monkeypatch.setattr(dailymed, "search_labels",
                        lambda drug, limit: [{"setid": drug, "title": drug.upper()}])

    def section(setid, title):
        passages = LISINOPRIL_PASSAGES if setid == "lisinopril" else []
        return Evidence(source="dailymed", title=title, url="u", text="",
                        meta={"setid": setid, "passages": [{"section": s, "text": t}
                                                            for s, t in passages]})

    monkeypatch.setattr(dailymed, "get_interactions_section", section)
    ev, _ = dailymed.find_interactions("meloxicam", "lisinopril",
                                       terms={"meloxicam": ["nsaid"]})
    assert len(ev) == 1 and ev[0].meta["match"] == "class"
    assert ev[0].meta["matched_term"] == "nsaid" and ev[0].meta["label_drug"] == "lisinopril"
    # the LLM is told the drug itself is not named in this fragment
    assert "найден по классу «nsaid»" in common.describe(ev[0])
