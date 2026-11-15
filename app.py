"""Streamlit UI for the DDI multi-agent app."""
import streamlit as st

from ddi.agents.orchestrator import run

st.set_page_config(page_title="DDI Agents", page_icon="💊", layout="centered")

st.title("💊 Взаимодействие лекарств")
st.caption("Мультиагентный сбор информации из DailyMed (инструкции FDA) и PubMed (литература).")
st.warning("Справочный инструмент, не медицинская консультация. "
           "Решения о приёме лекарств принимайте с врачом или фармацевтом.")

with st.form("query"):
    col1, col2 = st.columns(2)
    drug_a = col1.text_input("Препарат 1 (англ.)", placeholder="warfarin")
    drug_b = col2.text_input("Препарат 2 (англ.)", placeholder="ibuprofen")
    submitted = st.form_submit_button("Проверить")

if submitted:
    if not drug_a.strip() or not drug_b.strip():
        st.error("Укажите оба препарата.")
        st.stop()

    with st.spinner("Агенты собирают информацию…"):
        report = run(drug_a, drug_b)

    st.subheader("Итог")
    st.write(report.answer)

    with st.expander("Проверка критиком"):
        st.write(report.critique)

    for r in report.reports:
        with st.expander(f"Источник: {r.source} · фрагментов: {len(r.evidence)}"):
            for err in r.errors:
                st.error(err)
            for ev in r.evidence:
                st.markdown(f"**[{ev.title}]({ev.url})**")
                st.write(ev.text)
