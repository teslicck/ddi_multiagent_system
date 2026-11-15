"""LLM agents with a scripted fake LLM — no network, no API key."""
from __future__ import annotations

import pytest

from ddi import llm
from ddi.agents import common, critic, orchestrator, retrievers, synthesizer
from ddi.models import Evidence, SourceReport
from ddi.sources import dailymed, pubmed


def dm(text="Ibuprofen increases bleeding risk with warfarin.", **meta):
    return Evidence(source="dailymed", title="WARFARIN TABLET — 7.3 Bleeding",
                    url="https://dailymed.example/w1", text=text, meta=meta)


def pm(pmid="111", text="Ibuprofen plus warfarin raised INR.", **meta):
    return Evidence(source="pubmed", title=f"Paper {pmid}", url=f"https://pubmed.example/{pmid}/",
                    text=text, meta={"pmid": pmid, "year": "2014",
                                     "pub_types": ["Case Reports"], **meta})


class FakeLLM:
    """Records calls; answers by matching the system prompt."""

    def __init__(self, **replies):
        self.replies = replies            # key fragment of system prompt -> reply / Exception
        self.calls: list[dict] = []

    def __call__(self, system, user, *, temperature=0.0, max_tokens=800, model=None):
        self.calls.append({"system": system, "user": user, "model": model,
                           "max_tokens": max_tokens})
        for key, reply in self.replies.items():
            if key in system:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"unexpected prompt: {system[:80]}")


@pytest.fixture(autouse=True)
def no_rxnav(monkeypatch):
    """Orchestrator tests here are about agents; names pass through unchanged."""
    from ddi.models import DrugName
    from ddi.sources import rxnorm
    monkeypatch.setattr(rxnorm, "normalize", lambda s: DrugName(
        input=s.strip(), name=s.strip().lower(), status="exact"))


@pytest.fixture
def fake_llm(monkeypatch):
    def install(**replies):
        fake = FakeLLM(**replies)
        monkeypatch.setattr(llm, "complete", fake)
        monkeypatch.setattr(llm, "critic_model", lambda: "critic-model")
        return fake
    return install


# --------------------------------------------------------------------------- common

def test_refs_and_citation_parsing():
    evs = common.assign_refs([dm(), pm("42"), dm()])
    assert [e.ref for e in evs] == ["DM1", "PMID:42", "DM2"]
    text = "A [DM1]. B [PMID: 42][dm2]. C [DM1, PMID:42]. Not a ref [1] [see above]."
    assert common.citations(text) == ["DM1", "PMID:42", "DM2"]


def test_lookalike_brackets_from_gpt_oss_are_normalized(fake_llm):
    # live case: gpt-oss answered with 【DM1】【PMID:2783873】
    evs = common.assign_refs([dm(), pm("2783873")])
    raw = "**Вывод:** риск кровотечения повышен【DM1】【PMID:2783873】."
    assert common.citations(raw) == ["DM1", "PMID:2783873"]
    assert common.citation_problems(common.normalize_citations(raw), evs) == []

    fake_llm(**{"синтезатор": raw})
    answer, _ = synthesizer.synthesize("warfarin", "ibuprofen", reports())
    assert answer == "**Вывод:** риск кровотечения повышен[DM1][PMID:2783873]."


def test_is_not_found_tolerates_punctuation():
    assert common.is_not_found("НЕ НАЙДЕНО.")
    assert common.is_not_found("**не найдено**")
    assert not common.is_not_found("- Риск кровотечения [DM1]")


def test_evidence_block_respects_budget_keeps_first():
    evs = common.assign_refs([pm(str(i), text="x" * 1000) for i in range(10)])
    block = common.evidence_block(evs, max_chars=2500)
    assert block.startswith("[PMID:0] Paper 0 (2014; Case Reports)")
    assert "[PMID:1]" in block and "[PMID:3]" not in block


def test_citation_problems():
    evs = common.assign_refs([dm(), pm("42")])
    answer = ("**Вывод:** риск кровотечения повышен [DM1][PMID:42].\n"
              "**Что происходит:** усиление антикоагулянтного эффекта у всех пациентов.\n"
              "Ссылка на несуществующее [PMID:999].\n"
              "По PubMed данных не найдено.\n"
              "**Ограничения:**")
    problems = common.citation_problems(answer, evs)
    assert any("[PMID:999]" in p for p in problems)
    assert any("Утверждение без ссылки" in p and "усиление" in p for p in problems)
    assert not any("не найдено" in p for p in problems)          # allowed without a ref
    assert len(problems) == 2


def test_parse_json_object_handles_fences_and_prose():
    raw = 'Вот результат:\n```json\n{"verdict": "issues", "issues": [{"claim": "a {b}", "problem": "c"}]}\n```'
    assert common.parse_json_object(raw)["issues"][0]["claim"] == "a {b}"
    assert common.parse_json_object("no json here") is None
    assert common.parse_json_object("{broken {\"ok\": 1}") == {"ok": 1}


# --------------------------------------------------------------------------- retrievers

def test_summary_skipped_when_no_evidence(fake_llm):
    fake = fake_llm()
    summary, errors = retrievers.summarize(retrievers.DAILYMED_SYSTEM, "a", "b", [])
    assert summary == common.NOT_FOUND_MARK and errors == [] and fake.calls == []


def test_summary_prompt_contains_refs_and_drugs(fake_llm):
    fake = fake_llm(**{"DailyMed": "- Риск кровотечения [DM1]"})
    evs = common.assign_refs([dm()])
    summary, errors = retrievers.summarize(retrievers.DAILYMED_SYSTEM, "warfarin", "ibuprofen", evs)
    assert summary == "- Риск кровотечения [DM1]" and errors == []
    call = fake.calls[0]
    assert "warfarin и ibuprofen" in call["system"] and "[DM1]" in call["system"]
    assert "[DM1] WARFARIN TABLET" in call["user"]
    assert "{" not in call["system"]                   # all placeholders filled


def test_summary_llm_failure_keeps_evidence(fake_llm, monkeypatch):
    fake_llm(**{"DailyMed": llm.LLMNotConfigured("LLM_API_KEY is empty.")})
    monkeypatch.setattr(dailymed, "find_interactions", lambda a, b, per_direction: ([dm()], []))
    report = retrievers.dailymed_agent("warfarin", "ibuprofen")
    assert report.evidence[0].ref == "DM1" and report.summary == ""
    assert "LLM_API_KEY is empty" in report.errors[0]


def test_pubmed_agent_refs_are_pmids(fake_llm, monkeypatch):
    fake_llm(**{"PubMed": "- Случай [PMID:7]"})
    monkeypatch.setattr(pubmed, "find_interactions", lambda a, b, keep: ([pm("7")], []))
    report = retrievers.pubmed_agent("warfarin", "ibuprofen")
    assert report.evidence[0].ref == "PMID:7" and report.summary == "- Случай [PMID:7]"


# --------------------------------------------------------------------------- synthesizer

def reports(dm_summary="- Риск кровотечения [DM1]", pm_summary="- Случай [PMID:111]"):
    d = SourceReport(source="dailymed", query="q", evidence=common.assign_refs([dm()]),
                     summary=dm_summary)
    p = SourceReport(source="pubmed", query="q", evidence=common.assign_refs([pm()]),
                     summary=pm_summary)
    return [d, p]


def test_no_evidence_no_llm_and_never_says_no_interaction(fake_llm):
    fake = fake_llm()
    empty = [SourceReport(source="dailymed", query="q", summary=common.NOT_FOUND_MARK),
             SourceReport(source="pubmed", query="q", summary=common.NOT_FOUND_MARK)]
    answer, errors = synthesizer.synthesize("metformin", "amoxicillin", empty)
    assert "не найдено" in answer and "не означает, что взаимодействия нет" in answer
    assert fake.calls == [] and errors == []


def test_agents_found_nothing_relevant_skips_llm(fake_llm):
    fake = fake_llm()
    answer, _ = synthesizer.synthesize("a", "b", reports("НЕ НАЙДЕНО", "НЕ НАЙДЕНО."))
    assert "не найдено" in answer and fake.calls == []


def test_synthesizer_gets_summaries_and_not_found_marker(fake_llm):
    fake = fake_llm(**{"синтезатор": "**Вывод:** риск [DM1]."})
    answer, errors = synthesizer.synthesize("warfarin", "ibuprofen",
                                            reports(pm_summary="НЕ НАЙДЕНО"))
    assert answer == "**Вывод:** риск [DM1]." and errors == []
    user = fake.calls[0]["user"]
    assert "### DailyMed" in user and "- Риск кровотечения [DM1]" in user
    assert "### PubMed (научные статьи)\nНЕ НАЙДЕНО" in user


def test_synthesizer_uses_raw_fragments_when_summary_failed(fake_llm):
    fake = fake_llm(**{"синтезатор": "ok [DM1]"})
    synthesizer.synthesize("warfarin", "ibuprofen", reports(dm_summary=""))
    assert "Сводка не получена" in fake.calls[0]["user"]
    assert "Ibuprofen increases bleeding risk" in fake.calls[0]["user"]


def test_synthesizer_fallback_lists_sources(fake_llm):
    fake_llm(**{"синтезатор": RuntimeError("LLM error 500")})
    answer, errors = synthesizer.synthesize("warfarin", "ibuprofen", reports())
    assert "[DM1]" in answer and "[PMID:111]" in answer
    assert errors == ["Синтезатор: LLM error 500"]


# --------------------------------------------------------------------------- critic

def test_critic_ok_uses_critic_model_and_cited_fragments_only(fake_llm):
    fake = fake_llm(**{"критик": '{"verdict": "ok", "issues": []}'})
    rs = reports()
    rs[1].evidence.append(common.assign_refs([pm("222", text="UNCITED TEXT")])[0])
    summary, issues, errors = critic.critique("**Вывод:** риск кровотечения повышен [DM1].", rs)
    assert issues == [] and errors == []
    assert summary.startswith("Все утверждения подтверждаются")
    call = fake.calls[0]
    assert call["model"] == "critic-model"
    assert "Ibuprofen increases bleeding" in call["user"] and "UNCITED TEXT" not in call["user"]


def test_critic_merges_llm_and_deterministic_issues(fake_llm):
    fake_llm(**{"критик": 'Результат: {"verdict": "issues", "issues": ['
                          '{"claim": "у всех пациентов", "problem": "во фрагменте этого нет"}]}'})
    answer = "**Вывод:** риск кровотечения у всех пациентов [DM1].\nЕщё ссылка [PMID:999]."
    summary, issues, errors = critic.critique(answer, reports())
    assert summary == "Найдено замечаний: 2."
    assert any("[PMID:999]" in i for i in issues)
    assert "«у всех пациентов»: во фрагменте этого нет" in issues


def test_critic_bad_json_and_failure_are_errors_not_crashes(fake_llm):
    fake_llm(**{"критик": "Всё хорошо!"})
    _, issues, errors = critic.critique("**Вывод:** риск кровотечения повышен [DM1].", reports())
    assert issues == [] and errors == ["Критик вернул ответ не в формате JSON"]

    fake_llm(**{"критик": llm.LLMRateLimited("429")})
    summary, _, errors = critic.critique("**Вывод:** риск кровотечения повышен [DM1].", reports())
    assert "LLM-проверка не выполнена" in summary and errors[0].startswith("Критик:")


def test_critic_skips_without_evidence(fake_llm):
    fake = fake_llm()
    summary, issues, errors = critic.critique("не найдено", [SourceReport(source="pubmed", query="q")])
    assert "не требуется" in summary and fake.calls == []


def test_critic_skips_not_found_answer_even_with_evidence(fake_llm):
    # metformin + amoxicillin live case: PubMed returned 1 article, agent said NOT FOUND
    fake = fake_llm()
    answer = common.not_found_answer("metformin", "amoxicillin")
    summary, issues, errors = critic.critique(answer, reports())
    assert "не требуется" in summary and issues == [] and fake.calls == []


# --------------------------------------------------------------------------- end to end

def test_orchestrator_end_to_end_with_four_llm_calls(fake_llm, monkeypatch):
    monkeypatch.setattr(dailymed, "find_interactions", lambda a, b, per_direction: ([dm()], []))
    monkeypatch.setattr(pubmed, "find_interactions", lambda a, b, keep: ([pm()], []))
    fake = fake_llm(**{
        "агент источника DailyMed": "- Риск кровотечения [DM1]",
        "агент источника PubMed": "- Случай повышения МНО [PMID:111]",
        "синтезатор": "**Вывод:** риск кровотечения повышен [DM1][PMID:111].",
        "критик": '{"verdict": "ok", "issues": []}',
    })
    report = orchestrator.run(" warfarin ", "ibuprofen")
    assert report.answer.startswith("**Вывод:**")
    assert report.issues == [] and report.errors == []
    assert len(fake.calls) == 4                        # budget from CLAUDE.md
    assert [r.summary for r in report.reports] == ["- Риск кровотечения [DM1]",
                                                   "- Случай повышения МНО [PMID:111]"]


def test_orchestrator_nothing_relevant_uses_one_llm_call(fake_llm, monkeypatch):
    monkeypatch.setattr(dailymed, "find_interactions", lambda a, b, per_direction: ([], []))
    monkeypatch.setattr(pubmed, "find_interactions", lambda a, b, keep: ([pm()], []))
    fake = fake_llm(**{"агент источника PubMed": "НЕ НАЙДЕНО"})
    report = orchestrator.run("metformin", "amoxicillin")
    assert common.is_not_found_answer(report.answer)
    assert len(fake.calls) == 1                        # only the PubMed summary
