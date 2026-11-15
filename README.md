# DDI Agents

Мультиагентное приложение: собирает информацию о взаимодействии двух лекарств
из **DailyMed** (инструкции FDA, раздел 7 «Drug Interactions») и **PubMed**
(научные статьи), сводит её в ответ со ссылками и проверяет агентом-критиком.

> Справочный инструмент, не медицинская консультация.

## Архитектура

```
Пользователь (Streamlit)
        │
   Оркестратор ──┬── DailyMed-агент ── DailyMed REST API
                 └── PubMed-агент  ── NCBI E-utilities
        │
   Синтезатор (LLM) → Критик (LLM) → ответ со ссылками
```

## Структура

```
app.py                  Streamlit-интерфейс
ddi/config.py           настройки (st.secrets → env → .env)
ddi/models.py           Evidence, SourceReport, FinalReport
ddi/llm.py              обёртка над LLM (провайдер выберем позже)
ddi/sources/            клиенты DailyMed и PubMed (без LLM)
ddi/agents/             retrievers, synthesizer, critic, orchestrator
tests/                  тесты
```

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # заполнить ключи
streamlit run app.py
pytest
```

## Деплой (Streamlit Community Cloud)

1. Запушить репозиторий на GitHub (без `secrets.toml`).
2. share.streamlit.io → New app → репозиторий, ветка `main`, файл `app.py`.
3. Settings → Secrets: вставить содержимое `secrets.toml`.
