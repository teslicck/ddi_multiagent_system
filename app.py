"""Streamlit UI for the DDI multi-agent app."""
import re

import streamlit as st

from ddi.agents.common import is_not_found
from ddi.agents.orchestrator import run
from ddi.models import FinalReport

st.set_page_config(page_title="DDI Agents", page_icon="💊", layout="centered")

SOURCE_TITLES = {"dailymed": "DailyMed — инструкции FDA", "pubmed": "PubMed — научные статьи"}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_run(drug_a: str, drug_b: str) -> FinalReport:
    # Same pair -> same answer for a day; saves the Groq quota (8000 tokens/min).
    return run(drug_a, drug_b)


def link_citations(text: str, report: FinalReport) -> str:
    """[DM1] / [PMID:123] -> Markdown links to the source."""
    urls = {ev.ref.upper(): ev.url for r in report.reports for ev in r.evidence}

    def repl(m: re.Match) -> str:
        ref = re.sub(r"\s+", "", m.group(1)).upper()
        return f"[\\[{ref}\\]]({urls[ref]})" if ref in urls else m.group(0)

    return re.sub(r"\[(DM\d+|PMID:\s?\d+)\]", repl, text, flags=re.I)


def show_normalization(report: FinalReport) -> None:
    """What each typed name was searched as (RxNorm)."""
    for d in report.drugs:
        if d.status == "approximate":
            st.warning(f"«{d.input}» не найдено точно; распознано как **{d.name}** "
                       f"(RxNorm: {d.matched}). Проверьте, что это нужный препарат.")
        elif d.status == "unavailable":
            st.warning(f"RxNorm недоступен: «{d.input}» ищется как введено, "
                       f"торговые названия и опечатки не распознаются.")
        elif d.changed:
            st.caption(f"«{d.input}» → действующее вещество **{d.name}** (RxNorm)")
        if d.class_error:
            st.warning(f"Классы препарата {d.name} не получены ({d.class_error}): инструкции "
                       f"ищутся только по названию, упоминания класса (например, «NSAIDs») "
                       f"могут быть пропущены.")
    classes = [f"{d.name}: {', '.join(d.classes)}" for d in report.drugs if d.classes]
    if classes:
        st.caption("Классы FDA (RxClass): " + " · ".join(classes))


st.title("💊 Взаимодействие лекарств")
st.caption("Мультиагентный сбор информации из DailyMed (инструкции FDA) и PubMed (литература).")
st.warning("Справочный инструмент, не медицинская консультация. "
           "Решения о приёме лекарств принимайте с врачом или фармацевтом.")

with st.form("query"):
    col1, col2 = st.columns(2)
    drug_a = col1.text_input("Препарат 1 (англ., МНН или торговое)", placeholder="warfarin")
    drug_b = col2.text_input("Препарат 2 (англ., МНН или торговое)", placeholder="Advil")
    submitted = st.form_submit_button("Проверить")

if submitted:
    a, b = drug_a.strip(), drug_b.strip()      # RxNorm handles case; keep it for messages
    if not a or not b:
        st.error("Укажите оба препарата.")
        st.stop()
    if a.lower() == b.lower():
        st.error("Укажите два разных препарата.")
        st.stop()

    with st.spinner("Агенты собирают информацию: DailyMed и PubMed, затем синтез, проверка "
                    "и исправление по замечаниям (обычно 20–40 с)…"):
        try:
            report = cached_run(a, b)
        except Exception as exc:          # last-resort guard: never show a traceback
            st.error(f"Не удалось выполнить запрос: {exc}")
            st.stop()

    if report.stopped:                    # unknown name / combination / same drug
        st.error(report.answer)
        st.stop()

    show_normalization(report)
    st.subheader("Итог")
    if report.revision_note:
        (st.success if report.revised else st.info)(report.revision_note)
    st.markdown(link_citations(report.answer, report))
    for err in report.errors:
        st.warning(err)

    stage = "Повторная проверка" if report.revised else "Проверка критиком"
    label = f"✅ {stage}" if not report.issues else f"⚠️ {stage}: замечаний {len(report.issues)}"
    with st.expander(label, expanded=bool(report.issues)):
        st.write(report.critique)
        for issue in report.issues:
            st.markdown(f"- {link_citations(issue, report)}")

    if report.draft:
        with st.expander(f"Исходный ответ и замечания к нему ({len(report.draft_issues)})"):
            for issue in report.draft_issues:
                st.markdown(f"- {link_citations(issue, report)}")
            st.divider()
            st.markdown(link_citations(report.draft, report))

    for r in report.reports:
        title = SOURCE_TITLES.get(r.source, r.source)
        with st.expander(f"{title} · фрагментов: {len(r.evidence)}"):
            for err in r.errors:
                st.error(err)
            if r.summary and not is_not_found(r.summary):
                st.markdown("**Сводка агента**")
                st.markdown(link_citations(r.summary, report))
                st.divider()
            elif r.evidence:
                st.info("Агент не нашёл во фрагментах сведений именно об этой паре.")
            for ev in r.evidence:
                meta = ev.meta
                extra = f" · {meta['year']}" if meta.get("year") else ""
                st.markdown(f"**[{ev.ref}] [{ev.title}]({ev.url})**{extra}")
                st.caption(ev.text)
            st.caption(f"Запрос: `{r.query}`")
