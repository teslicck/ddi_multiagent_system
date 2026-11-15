"""DailyMed client tests on saved (trimmed) API responses — no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ddi.sources import dailymed

FIXTURES = Path(__file__).parent / "fixtures" / "dailymed"
PLR = (FIXTURES / "label_plr.xml").read_bytes()
OLD = (FIXTURES / "label_old_format.xml").read_bytes()


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def json(self):
        return json.loads(self.content)


@pytest.fixture
def fake_api(monkeypatch):
    """Route requests.get to fixtures; returns the list of requested URLs."""
    routes: dict[str, FakeResponse] = {}
    calls: list[tuple[str, dict | None]] = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        key = url.removeprefix(dailymed.BASE_URL + "/")
        if key == "spls.json":
            key = f"spls.json:{params['drug_name']}"
        if key not in routes:
            return FakeResponse(404)
        return routes[key]

    monkeypatch.setattr(dailymed.requests, "get", fake_get)
    monkeypatch.setattr(dailymed._get.retry, "sleep", lambda _: None)
    fake_get.routes = routes
    fake_get.calls = calls
    return fake_get


def search(*items):
    return FakeResponse(content=json.dumps({"data": list(items)}).encode())


# --------------------------------------------------------------------------- parsing

def test_plr_label_uses_loinc_section_and_skips_highlights():
    _, passages = dailymed.parse_interactions(PLR)
    texts = [t for _, t in passages]
    assert any("pharmacodynamic" in t for t in texts)
    assert not any("must be skipped" in t for t in texts)          # excerpt
    assert not any("not the interactions section" in t for t in texts)  # section 5
    rows = [(s, t) for s, t in passages if "celecoxib" in t]
    assert rows == [("7.3 Drugs that Increase Bleeding Risk",
                     "Nonsteroidal Anti-Inflammatory Agents | celecoxib, diclofenac, ibuprofen, naproxen")]


def test_table_header_rows_group_following_rows():
    _, passages = dailymed.parse_interactions(PLR)
    groups = [t for s, t in passages if s.startswith("7.4")]
    assert groups == [
        "Strong CYP3A4 inhibitors / Clinical Impact: | Increased exposure. / "
        "Examples: | clarithromycin, ketoconazole",
        "Grapefruit juice / Intervention: | Avoid large quantities.",
    ]
    hits = dailymed.filter_passages(passages, "clarithromycin")
    assert hits[0][1].startswith("Strong CYP3A4 inhibitors")


def test_old_label_matched_by_title_and_lab_section_skipped():
    title, passages = dailymed.parse_interactions(OLD)
    assert title == "Ibuprofen Tablets, USP Rx only"
    sections = {s for s, _ in passages}
    assert "Warfarin-type anticoagulants" in sections
    assert "Methotrexate" in sections
    assert not any("lab section" in t for _, t in passages)
    # the mis-coded "Preexisting asthma" section is included but harmless after filtering
    assert dailymed.filter_passages(passages, "methotrexate")[0][0] == "Methotrexate"


def test_filter_matches_section_title_and_class_terms():
    _, passages = dailymed.parse_interactions(OLD)
    hits = dailymed.filter_passages(passages, "warfarin")
    assert [(s, term) for s, _, term in hits] == [("Warfarin-type anticoagulants", "warfarin")]

    _, plr = dailymed.parse_interactions(PLR)
    hits = dailymed.filter_passages(plr, "ibuprofen")
    assert hits[0][2] == "ibuprofen"                 # exact name ranks first


def test_filter_does_not_match_inside_other_words():
    passages = [("X", "Nonaspirin products"), ("Y", "aspirin increases bleeding")]
    assert [s for s, _, _ in dailymed.filter_passages(passages, "aspirin")] == ["Y"]


def test_unknown_drug_has_no_class_terms():
    assert dailymed.terms_for("  Zolpidem ") == ["zolpidem"]


# --------------------------------------------------------------------------- HTTP

def test_search_prefers_single_ingredient_rx_labels(fake_api):
    fake_api.routes["spls.json:ibuprofen"] = FakeResponse(
        content=(FIXTURES / "spls_ibuprofen.json").read_bytes())
    assert dailymed.find_label_setids(" Ibuprofen ") == ["ibu-setid", "combo-setid"]
    _, params = fake_api.calls[0]
    assert params["doctype"] == dailymed.RX_DOCTYPE


def test_find_interactions_both_directions(fake_api):
    fake_api.routes.update({
        "spls.json:warfarin": search({"setid": "w1", "title": "WARFARIN SODIUM TABLET [X]"}),
        "spls.json:ibuprofen": search({"setid": "i1", "title": "IBUPROFEN TABLET [Y]"}),
        "spls/w1.xml": FakeResponse(content=PLR),
        "spls/i1.xml": FakeResponse(content=OLD),
    })
    evidence, errors = dailymed.find_interactions("warfarin", "ibuprofen")
    assert errors == []
    by_label = {e.meta["label_drug"]: e for e in evidence}
    assert set(by_label) == {"warfarin", "ibuprofen"}
    w = by_label["warfarin"]
    assert w.title == "WARFARIN SODIUM TABLET [X] — 7.3 Drugs that Increase Bleeding Risk"
    assert w.url.endswith("setid=w1")
    assert w.meta["about"] == "ibuprofen"
    assert by_label["ibuprofen"].meta["section"] == "Warfarin-type anticoagulants"


def test_falls_back_to_next_label_and_reports_failures(fake_api):
    fake_api.routes.update({
        "spls.json:warfarin": search({"setid": "bad", "title": "WARFARIN A"},
                                     {"setid": "w1", "title": "WARFARIN B"}),
        "spls.json:ibuprofen": search(),
        "spls/w1.xml": FakeResponse(content=PLR),
        # "spls/bad.xml" -> 404
    })
    evidence, errors = dailymed.find_interactions("warfarin", "ibuprofen")
    assert evidence and all(e.meta["label_drug"] == "warfarin" for e in evidence)
    assert any("bad" in e for e in errors)
    assert any("ibuprofen не найдена" in e for e in errors)


def test_retries_transient_errors_then_reports(fake_api, monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(dailymed.requests, "get", boom)
    evidence, errors = dailymed.find_interactions("warfarin", "ibuprofen")
    assert evidence == []
    assert len(errors) == 2 and all("не удался" in e for e in errors)


def test_long_passages_are_clipped():
    text = "word " * 1000
    clipped = dailymed._clip(text)
    assert len(clipped) <= dailymed.MAX_PASSAGE_CHARS + 2 and clipped.endswith("…")


def test_clip_keeps_the_matched_drug_far_into_a_table():
    text = ("Drugs Affected By Clarithromycin / " + "Other drug | text. " * 300
            + "Simvastatin | Contraindicated with clarithromycin. / " + "More | rows. " * 100)
    clipped = dailymed._clip(text, "simvastatin")
    assert len(clipped) <= dailymed.MAX_PASSAGE_CHARS
    assert clipped.startswith("Drugs Affected By Clarithromycin / … ")
    assert "Simvastatin | Contraindicated" in clipped
    assert clipped.endswith("…")
