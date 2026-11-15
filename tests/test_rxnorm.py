"""RxNorm normalization on saved real RxNav responses — no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ddi import llm
from ddi.agents import orchestrator
from ddi.models import DrugName
from ddi.sources import dailymed, pubmed, rxnorm

FIXTURES = Path(__file__).parent / "fixtures" / "rxnorm"


def fx(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


EMPTY_EXACT = {"idGroup": {}}

# (path, distinguishing param) -> response; built from real RxNav replies
ROUTES = {
    ("rxcui.json", "Advil"): fx("exact_advil.json"),
    ("rxcui/153010/related.json", None): fx("related_153010.json"),
    ("rxcui/153010/properties.json", None): fx("props_153010.json"),
    ("rxcui.json", "warfarn"): fx("exact_warfarn.json"),
    ("approximateTerm.json", "warfarn"): fx("approx_warfarn.json"),
    ("rxcui/11289/related.json", None): fx("related_11289.json"),
    ("rxcui.json", "Augmentin"): {"idGroup": {"rxnormId": ["151392"]}},
    ("rxcui/151392/related.json", None): fx("related_151392.json"),
    ("rxcui/151392/properties.json", None): {"properties": {"name": "Augmentin"}},
    ("rxcui.json", "xyzabc"): EMPTY_EXACT,
    ("approximateTerm.json", "xyzabc"): fx("approx_xyzabc.json"),
    ("rxcui.json", "Vicodin"): EMPTY_EXACT,
    ("approximateTerm.json", "Vicodin"): {"approximateGroup": {"candidate": [
        {"rxcui": "128793", "rank": "1", "source": "RXNORM", "name": "Vicodin"}]}},
    ("rxcui/128793/related.json", None): fx("related_128793.json"),
    ("rxcui.json", "ibuprofen"): {"idGroup": {"rxnormId": ["5640"]}},
    ("rxcui/5640/related.json", None): {"relatedGroup": {"conceptGroup": [
        {"tty": "IN", "conceptProperties": [{"rxcui": "5640", "name": "ibuprofen"}]}]}},
    ("rxcui/5640/properties.json", None): {"properties": {"name": "ibuprofen"}},
}


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self._data, self.status_code = data, status_code

    def json(self):
        return self._data


@pytest.fixture
def fake_rxnav(monkeypatch):
    calls = []
    # classes are tested in test_rxclass.py; here only make sure normalize passes them on
    monkeypatch.setattr(rxnorm.rxclass, "class_terms",
                        lambda rxcui: (["nsaid"], ["Nonsteroidal Anti-inflammatory Drug"], None)
                        if rxcui == "5640" else ([], [], None))

    def fake_get(url, params=None, timeout=None):
        path = url.removeprefix(rxnorm.BASE_URL + "/")
        key = (params or {}).get("name") or (params or {}).get("term")
        calls.append((path, key))
        if (path, key) in ROUTES:
            return FakeResponse(ROUTES[(path, key)])
        if (path, None) in ROUTES:
            return FakeResponse(ROUTES[(path, None)])
        return FakeResponse(status_code=404)

    monkeypatch.setattr(rxnorm.requests, "get", fake_get)
    monkeypatch.setattr(rxnorm._get.retry, "sleep", lambda _: None)
    return calls


# --------------------------------------------------------------------------- normalize

def test_brand_name_to_ingredient(fake_rxnav):
    d = rxnorm.normalize(" Advil ")
    assert (d.input, d.name, d.status, d.rxcui, d.matched) == \
           ("Advil", "ibuprofen", "exact", "153010", "Advil")
    assert d.changed
    # classes are looked up for the ingredient (5640), not the brand concept (153010)
    assert d.ingredient_rxcui == "5640" and d.class_terms == ["nsaid"]
    assert d.classes == ["Nonsteroidal Anti-inflammatory Drug"] and d.class_error == ""


def test_ingredient_stays_and_is_not_marked_changed(fake_rxnav):
    d = rxnorm.normalize("ibuprofen")
    assert d.name == "ibuprofen" and d.status == "exact" and not d.changed


def test_typo_uses_approximate_match_with_rxnorm_name(fake_rxnav):
    d = rxnorm.normalize("warfarn")
    assert (d.name, d.status, d.matched) == ("warfarin", "approximate", "warfarin")
    # the first rank-1 candidate in the real reply has no name (source GS); we take RXNORM's
    assert ("approximateTerm.json", "warfarn") in fake_rxnav


def test_combination_product_is_flagged(fake_rxnav):
    d = rxnorm.normalize("Augmentin")
    assert d.status == "combination" and d.ingredients == ["clavulanate", "amoxicillin"]


def test_unknown_word_not_found(fake_rxnav):
    d = rxnorm.normalize("xyzabc")
    assert d.status == "not_found" and d.name == "xyzabc"


def test_obsolete_concept_without_ingredients_not_found(fake_rxnav):
    # real case: Vicodin's RxNorm concept is obsolete since 2022, no IN relations
    assert rxnorm.normalize("Vicodin").status == "not_found"


def test_rxnav_down_falls_back_to_input(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(rxnorm.requests, "get", boom)
    monkeypatch.setattr(rxnorm._get.retry, "sleep", lambda _: None)
    d = rxnorm.normalize("Warfarin")
    assert (d.name, d.status) == ("warfarin", "unavailable")


# --------------------------------------------------------------------------- orchestrator

def drug(inp, name, status="exact", ingredients=None):
    return DrugName(input=inp, name=name, status=status, ingredients=ingredients or [name])


def test_input_problem_messages():
    ok = drug("Advil", "ibuprofen")
    assert orchestrator.input_problem(ok, drug("warfarin", "warfarin")) is None
    assert "не распознан" in orchestrator.input_problem(ok, drug("xyz", "xyz", "not_found"))
    combo = drug("Augmentin", "augmentin", "combination", ["clavulanate", "amoxicillin"])
    assert "clavulanate + amoxicillin" in orchestrator.input_problem(combo, ok)
    assert "один и тот же" in orchestrator.input_problem(ok, drug("ibuprofen", "ibuprofen"))
    # RxNav down is not a reason to stop
    assert orchestrator.input_problem(drug("w", "w", "unavailable"), ok) is None


def test_unknown_name_stops_before_any_search_or_llm(fake_rxnav, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("must not be called")
    monkeypatch.setattr(dailymed, "find_interactions", forbidden)
    monkeypatch.setattr(pubmed, "find_interactions", forbidden)
    monkeypatch.setattr(llm, "complete", forbidden)
    report = orchestrator.run("Advil", "xyzabc")
    assert report.stopped and "«xyzabc» не распознан" in report.answer
    assert "не найдено данных" not in report.answer      # never the "no data" answer
    assert report.reports == []


def test_sources_are_searched_with_normalized_names(fake_rxnav, monkeypatch):
    seen = []
    monkeypatch.setattr(dailymed, "find_interactions",
                        lambda a, b, **kw: (seen.append(("dm", a, b)), ([], []))[1])
    monkeypatch.setattr(pubmed, "find_interactions",
                        lambda a, b, **kw: (seen.append(("pm", a, b)), ([], []))[1])
    def dm_find(a, b, **kw):
        seen.append(("dm", a, b))
        seen.append(("terms", kw.get("terms")))
        return [], []

    monkeypatch.setattr(dailymed, "find_interactions", dm_find)
    report = orchestrator.run("Advil", "warfarn")
    assert ("terms", {"ibuprofen": ["nsaid"], "warfarin": []}) in seen    # class terms passed on
    seen.remove(("terms", {"ibuprofen": ["nsaid"], "warfarin": []}))
    assert sorted(seen) == [("dm", "ibuprofen", "warfarin"), ("pm", "ibuprofen", "warfarin")]
    assert (report.drug_a, report.drug_b) == ("ibuprofen", "warfarin")
    assert [d.status for d in report.drugs] == ["exact", "approximate"]
    assert not report.stopped
