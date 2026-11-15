# DDI Agents

Мультиагентное приложение: собирает информацию о взаимодействии двух лекарств
из **DailyMed** (инструкции FDA: Boxed Warning, Contraindications, Drug Interactions) и **PubMed**
(научные статьи), сводит её в ответ со ссылками и проверяет агентом-критиком.

**Демо:** https://ddimultiagentsystemdaniiln.streamlit.app/

> Справочный инструмент, не медицинская консультация.

## Архитектура

```
Пользователь (Streamlit)
        │
   Нормализация названий (RxNorm: Advil → ibuprofen, опечатки, комбинации)
   + классы FDA (RxClass: meloxicam → NSAID, чтобы найти «NSAIDs» в инструкциях)
        │
   Оркестратор ──┬── DailyMed-агент ── DailyMed REST API  ─┐ параллельно,
                 └── PubMed-агент  ── NCBI E-utilities    ─┘ каждый с LLM-сводкой
        │
   Синтезатор (LLM) → Критик (другая LLM) → Редактор (исправляет по замечаниям)
        → повторная проверка → ответ со ссылками [DM1] / [PMID:…]
```

LLM: Groq (`openai/gpt-oss-120b`, критик — `qwen/qwen3.8-27b`), до 6 вызовов на запрос.

## Структура

```
app.py                  Streamlit-интерфейс
ddi/config.py           настройки (st.secrets → env → .env)
ddi/models.py           Evidence, SourceReport, FinalReport
ddi/llm.py              обёртка над LLM (OpenAI-совместимый API: Groq / OpenRouter / Gemini / Ollama)
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

## Оценка качества

`eval/` — эталонный набор из 40 пар (30 известных взаимодействий, 10 без) и скрипт оценки:

| Метрика | Значение |
|---|---|
| Поиск без LLM: взаимодействие найдено хотя бы в одном источнике | 30/30 |
| Полный конвейер: ответ сообщает о взаимодействии (выборка 10) | 10/10 |
| Ложные заявления о взаимодействии (10 пар без него) | 0/10 |
| Замечаний критика на ответ до → после исправления | 1,0 → 0,2 |

Подробности и разбор спорных случаев — `eval/README.md`.

## Деплой (Streamlit Community Cloud)

1. Запушить репозиторий на GitHub (без `secrets.toml`).
2. share.streamlit.io → New app → репозиторий, ветка `main`, файл `app.py`, Python 3.12.
3. Advanced settings → Secrets: достаточно двух строк
   ```toml
   LLM_PROVIDER = "groq"
   LLM_API_KEY = "gsk_..."
   ```
   Модели подставляются по умолчанию, ключ NCBI необязателен.

Каждый пуш в `main` пересобирает приложение автоматически. Приложение публичное:
все запросы тратят лимит одного ключа Groq (8 000 токенов в минуту на модель);
повторные запросы той же пары берутся из кэша на сутки.
