"""PubMed client tests on saved (trimmed) E-utilities responses — no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ddi.sources import pubmed

FIXTURES = Path(__file__).parent / "fixtures" / "pubmed"
ESEARCH = (FIXTURES / "esearch_warfarin_ibuprofen.json").read_bytes()
EFETCH = (FIXTURES / "efetch_warfarin_ibuprofen.xml").read_bytes()


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def json(self):
        return json.loads(self.content)


@pytest.fixture
def fake_api(monkeypatch):
    """Route requests.get by endpoint; records params of every call."""
    routes: dict[str, FakeResponse] = {"esearch.fcgi": FakeResponse(content=ESEARCH),
                                       "efetch.fcgi": FakeResponse(content=EFETCH)}
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=None):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append((endpoint, params))
        return routes.get(endpoint, FakeResponse(404))

    monkeypatch.setattr(pubmed.requests, "get", fake_get)
    monkeypatch.setattr(pubmed, "_throttle", lambda: None)
    monkeypatch.setattr(pubmed._get.retry, "sleep", lambda _: None)
    fake_get.routes, fake_get.calls = routes, calls
    return fake_get


# --------------------------------------------------------------------------- query / parsing

def test_query_matches_the_plan_and_strips_user_syntax():
    assert pubmed.build_query(" Warfarin ", "ibuprofen") == (
        "warfarin[tiab] AND ibuprofen[tiab] AND (drug interactions[mh] OR interaction[tiab])")
    assert "[tiab] OR" not in pubmed.build_query('warfarin[tiab] OR "x"', "b")


def test_parse_articles_fields():
    arts = {a["pmid"]: a for a in pubmed.parse_articles(EFETCH)}
    letter = arts["20043424"]
    assert letter["abstract"] == "" and letter["doi"] == "10.7748/en.17.8.7.s10"
    assert letter["journal"] == "Emerg Nurse" and "Letter" in letter["pub_types"]

    structured = arts["30782418"]
    assert structured["year"] == "1998"                       # from MedlineDate
    assert structured["journal"] == "Journal of clinical pharmacology"  # no ISO abbrev
    assert structured["abstract"].startswith("BACKGROUND: NSAIDs are widely used. RESULTS:")

    case = arts["11111111"]
    assert case["year"] == "2021"                             # ArticleDate fallback
    assert case["title"] == "Notes on ibuprofen: a case report."
    assert case["abstract"] == "A patient on warfarin also took ibuprofen last spring."


def test_scoring():
    arts = {a["pmid"]: a for a in pubmed.parse_articles(EFETCH)}
    score = lambda pmid: pubmed.score_article(arts[pmid], "warfarin", "ibuprofen")
    assert score("20043424") == 3      # both in title
    assert score("30782418") == 2      # one sentence + "increased ... bleeding"
    assert score("19719333") == 0      # different sentences only
    assert score("11111111") == 1      # co-mention without interaction words


def test_binding_site_co_mention_ranks_below_clinical_sentence():
    marker = {"title": "Albumin binding", "abstract":
              "Site markers warfarin and ibuprofen were used in the assay."}
    clinical = {"title": "NSAIDs", "abstract":
                "Ibuprofen increased bleeding in patients on warfarin."}
    # biophysics paper that also says "interaction" and "inhibited"
    biophys = {"title": "Ligand binding to human serum albumin", "abstract":
               "Its interaction with HSA inhibited the binding of warfarin and ibuprofen."}
    s = lambda art: pubmed.score_article(art, "warfarin", "ibuprofen")
    assert s(marker) == 1 and s(clinical) == 2 and s(biophys) == 1


def test_word_boundaries():
    art = {"title": "Prewarfarin era and ibuprofen", "abstract": ""}
    assert pubmed.score_article(art, "warfarin", "ibuprofen") == 0


# --------------------------------------------------------------------------- excerpt

def test_short_abstract_kept_whole():
    assert pubmed.excerpt("Short one.", "a", "b") == "Short one."


def test_long_abstract_keeps_pair_sentences_and_conclusion():
    filler = " ".join(f"Filler sentence number {i} about methods." for i in range(60))
    abstract = (f"Intro sentence. {filler} Warfarin with ibuprofen raised INR. {filler} "
                "CONCLUSIONS: Avoid the combination.")
    out = pubmed.excerpt(abstract, "warfarin", "ibuprofen")
    assert len(out) <= pubmed.MAX_ABSTRACT_CHARS
    assert "Warfarin with ibuprofen raised INR." in out
    assert out.endswith("CONCLUSIONS: Avoid the combination.")
    assert out.startswith("… ") and "Filler" not in out


# --------------------------------------------------------------------------- facade

def test_find_interactions_ranks_and_filters(fake_api):
    evidence, errors = pubmed.find_interactions("warfarin", "ibuprofen")
    assert errors == []
    # review (0) and bare co-mention (1) dropped; title hit ranks above the
    # clinical sentence even though esearch returned it second
    assert [e.meta["pmid"] for e in evidence] == ["20043424", "30782418"]
    assert [e.meta["score"] for e in evidence] == [3, 2]
    letter = evidence[0]
    assert letter.text == "Ibuprofen and warfarin."       # no abstract -> title as text
    assert letter.url == "https://pubmed.ncbi.nlm.nih.gov/20043424/"
    assert evidence[1].meta["year"] == "1998" and evidence[1].source == "pubmed"


def test_request_params_include_tool_and_optional_credentials(fake_api, monkeypatch):
    from ddi import config
    monkeypatch.setattr(pubmed, "get_settings",
                        lambda: config.Settings(ncbi_api_key="KEY", ncbi_email="me@example.org"))
    pubmed.find_interactions("warfarin", "ibuprofen")
    (ep1, p1), (ep2, p2) = fake_api.calls
    assert (ep1, ep2) == ("esearch.fcgi", "efetch.fcgi")
    assert p1["tool"] == "ddi-agents" and p1["api_key"] == "KEY" and p1["email"] == "me@example.org"
    assert p1["sort"] == "relevance" and p1["retmode"] == "json"
    assert p2["id"] == "19719333,20043424,30782418,11111111"


def test_no_credentials_means_no_empty_params(fake_api, monkeypatch):
    from ddi import config
    monkeypatch.setattr(pubmed, "get_settings", lambda: config.Settings(ncbi_api_key="", ncbi_email=""))
    pubmed.find_interactions("warfarin", "ibuprofen")
    _, params = fake_api.calls[0]
    assert "api_key" not in params and "email" not in params


def test_no_hits_skips_efetch(fake_api):
    fake_api.routes["esearch.fcgi"] = FakeResponse(
        content=json.dumps({"esearchresult": {"count": "0", "idlist": []}}).encode())
    assert pubmed.find_interactions("metformin", "amoxicillin") == ([], [])
    assert [ep for ep, _ in fake_api.calls] == ["esearch.fcgi"]


def test_errors_are_reported_not_raised(fake_api, monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(pubmed.requests, "get", boom)
    evidence, errors = pubmed.find_interactions("warfarin", "ibuprofen")
    assert evidence == [] and len(errors) == 1 and "PubMed" in errors[0]


def test_esearch_error_field(fake_api):
    fake_api.routes["esearch.fcgi"] = FakeResponse(
        content=json.dumps({"esearchresult": {"ERROR": "Invalid query"}}).encode())
    evidence, errors = pubmed.find_interactions("x", "y")
    assert evidence == [] and "Invalid query" in errors[0]


def test_throttle_spaces_calls(monkeypatch):
    from ddi import config
    monkeypatch.setattr(pubmed, "get_settings", lambda: config.Settings(ncbi_api_key=""))
    clock = {"t": 100.0}
    slept = []
    monkeypatch.setattr(pubmed.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: (slept.append(s), clock.update(t=clock["t"] + s)))
    monkeypatch.setattr(pubmed, "_last_call", 0.0)
    pubmed._throttle()
    pubmed._throttle()
    assert slept == [pytest.approx(0.34)]
