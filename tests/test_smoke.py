from ddi.agents.orchestrator import run
from ddi.sources import dailymed


def test_pipeline_runs_offline(monkeypatch):
    # keep the smoke test off the network
    monkeypatch.setattr(dailymed, "find_interactions", lambda a, b: ([], ["offline"]))
    report = run(" warfarin ", "ibuprofen")
    assert report.drug_a == "warfarin"
    assert {r.source for r in report.reports} == {"dailymed", "pubmed"}
    assert report.answer
