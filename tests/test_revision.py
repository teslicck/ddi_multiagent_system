"""Revision round (critic -> reviser -> critic again) with a scripted fake LLM."""
from __future__ import annotations

import pytest
import requests

from ddi import llm
from ddi.agents import orchestrator, reviser
from ddi.sources import dailymed, pubmed

from tests.test_agents import FakeLLM, dm, pm, reports

DRAFT = "**Вывод:** флуконазол — сильный ингибитор CYP3A4, сочетание противопоказано [DM1]."
FIXED = ("**Вывод:** инструкция говорит о классе сильных ингибиторов CYP3A4, флуконазол в нём "
         "не назван; риск миопатии повышен [DM1].")
ISSUE = '{"verdict": "issues", "issues": [{"claim": "сильный ингибитор", "problem": "во фрагменте не назван"}]}'
OK = '{"verdict": "ok", "issues": []}'
TWO = ('{"verdict": "issues", "issues": [{"claim": "a", "problem": "x"}, '
       '{"claim": "b", "problem": "y"}]}')

SUMMARIES = {"агент источника DailyMed": "- Риск [DM1]", "агент источника PubMed": "- Случай [PMID:111]",
             "синтезатор": DRAFT}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from ddi.models import DrugName
    from ddi.sources import rxnorm
    monkeypatch.setattr(rxnorm, "normalize", lambda s: DrugName(
        input=s.strip(), name=s.strip().lower(), status="exact"))
    monkeypatch.setattr(dailymed, "find_interactions",
                        lambda a, b, **kw: ([dm("Strong CYP3A4 inhibitors: ketoconazole.")], []))
    monkeypatch.setattr(pubmed, "find_interactions", lambda a, b, **kw: ([pm()], []))


@pytest.fixture
def install(monkeypatch):
    def _install(**replies):
        fake = FakeLLM(**{**SUMMARIES, **replies})
        monkeypatch.setattr(llm, "complete", fake)
        monkeypatch.setattr(llm, "critic_model", lambda: "critic-model")
        return fake
    return _install


def test_revision_accepted_when_recheck_is_clean(install):
    fake = install(**{"критик": [ISSUE, OK], "агент-редактор": FIXED})
    rep = orchestrator.run("fluconazole", "simvastatin")
    assert rep.revised and rep.answer == FIXED
    assert rep.draft == DRAFT and rep.draft_issues == ["«сильный ингибитор»: во фрагменте не назван"]
    assert rep.issues == [] and rep.errors == []
    assert "было 1, после повторной проверки 0" in rep.revision_note
    assert len(fake.calls) == 6                                   # 4 + reviser + re-check

    edit = next(c for c in fake.calls if "агент-редактор" in c["system"])
    assert edit["model"] is None                                  # main model, not the critic's
    assert "во фрагменте не назван" in edit["user"]               # the finding
    assert "Strong CYP3A4 inhibitors: ketoconazole." in edit["user"]   # ORIGINAL fragment
    assert DRAFT in edit["user"] and "{" not in edit["system"]
    recheck = [c for c in fake.calls if "критик" in c["system"]][1]
    assert FIXED in recheck["user"] and recheck["model"] == "critic-model"


def test_revision_rejected_when_recheck_finds_more(install):
    install(**{"критик": [ISSUE, TWO], "агент-редактор": FIXED})
    rep = orchestrator.run("fluconazole", "simvastatin")
    assert not rep.revised and rep.answer == DRAFT
    assert len(rep.issues) == 1 and "отклонён" in rep.revision_note
    assert rep.draft == DRAFT                                     # still shown for transparency


def test_reviser_failure_keeps_draft(install):
    install(**{"критик": [ISSUE], "агент-редактор": llm.LLMRateLimited("429")})
    rep = orchestrator.run("fluconazole", "simvastatin")
    assert not rep.revised and rep.answer == DRAFT and rep.draft == ""
    assert rep.errors == ["Редактор: 429"]
    assert "не получено" in rep.revision_note


def test_unverified_revision_is_shown_with_a_note(install):
    install(**{"критик": [ISSUE, RuntimeError("critic down")], "агент-редактор": FIXED})
    rep = orchestrator.run("fluconazole", "simvastatin")
    assert rep.revised and rep.answer == FIXED
    assert "повторная проверка не выполнена" in rep.revision_note
    assert rep.errors == ["Критик: critic down"]


def test_no_findings_no_revision(install):
    fake = install(**{"критик": OK})
    rep = orchestrator.run("fluconazole", "simvastatin")
    assert rep.answer == DRAFT and not rep.revised and rep.revision_note == ""
    assert not any("агент-редактор" in c["system"] for c in fake.calls)


def test_revise_normalizes_brackets_and_skips_without_issues(install):
    fake = install(**{"агент-редактор": "ok【DM1】"})
    assert reviser.revise("a", "b", DRAFT, [], reports()) == (DRAFT, [])
    assert fake.calls == []
    assert reviser.revise("a", "b", DRAFT, ["x"], reports()) == ("ok[DM1]", [])


# --------------------------------------------------------------------------- 429 handling

class Resp:
    def __init__(self, text="", headers=None):
        self.text, self.headers = text, headers or {}


def test_retry_after_from_header_and_groq_body():
    assert llm._retry_after(Resp(headers={"retry-after": "3"})) == 3.0
    body = "Rate limit reached ... Please try again in 7.36s. Need more tokens?"
    assert llm._retry_after(Resp(body)) == pytest.approx(7.36)
    assert llm._retry_after(Resp("try again in 1m2.5s")) == pytest.approx(62.5)
    assert llm._retry_after(Resp("no hint")) is None


def test_complete_waits_retry_after_then_succeeds(monkeypatch):
    from ddi import config
    monkeypatch.setattr(llm, "get_settings", lambda: config.Settings(
        llm_provider="groq", llm_api_key="k", llm_model="m"))
    replies = [(429, {"error": {"message": "Please try again in 1.5s."}}, {}),
               (200, {"choices": [{"message": {"content": "готово"}}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 3}}, {})]

    class R:
        def __init__(self, status, data, headers):
            self.status_code, self._data, self.headers = status, data, headers
            self.text = str(data)

        def json(self):
            return self._data

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R(*replies.pop(0)))
    slept = []
    monkeypatch.setattr(llm.complete.retry, "sleep", slept.append)
    llm.usage_log.clear()
    assert llm.complete("s", "u") == "готово"
    assert slept == [pytest.approx(2.0)]                         # 1.5 s hint + 0.5 s margin
    assert llm.usage_log[-1]["prompt"] == 10 and llm.usage_log[-1]["model"] == "m"
