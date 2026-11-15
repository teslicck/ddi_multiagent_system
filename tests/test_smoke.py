from ddi.agents.orchestrator import run


def test_pipeline_runs_with_stubs():
    report = run(" warfarin ", "ibuprofen")
    assert report.drug_a == "warfarin"
    assert {r.source for r in report.reports} == {"dailymed", "pubmed"}
    assert report.answer
