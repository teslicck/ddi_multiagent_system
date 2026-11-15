from ddi.agents.orchestrator import run
from ddi.sources import dailymed, pubmed


def test_pipeline_runs_offline(monkeypatch):
    # keep the smoke test off the network
    monkeypatch.setattr(dailymed, "find_interactions", lambda a, b, **kw: ([], ["offline"]))
    monkeypatch.setattr(pubmed, "find_interactions", lambda a, b, **kw: ([], ["offline"]))
    report = run(" warfarin ", "ibuprofen")
    assert report.drug_a == "warfarin"
    assert {r.source for r in report.reports} == {"dailymed", "pubmed"}
    # no evidence -> deterministic "not found" answer, no LLM needed
    assert "не найдено" in report.answer
