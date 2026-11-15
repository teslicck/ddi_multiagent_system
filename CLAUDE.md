# CLAUDE.md — DDI Agents

Сводка задачи для Claude Code. Читать перед любой работой в репозитории.

## Задача (учебный проект)

Курсовое задание: **мультиагентное приложение для сбора информации из внешних источников**.
Требования преподавателя:
- работает ровно с **2 источниками** из реестра курса (`Источники.pdf`, реестр медицинских источников);
- обёрнуто в **web-интерфейс** (Streamlit) и **выложено в web** (Streamlit Community Cloud);
- к семинару нужны доклад и презентация (архитектура, взаимодействие агентов, демо, ограничения).

**Тема:** взаимодействие лекарств (drug-drug interaction, DDI). Пользователь вводит два препарата
и получает сводку о взаимодействии со ссылками на источники.

## Принятые решения

| Вопрос | Решение | Почему |
|---|---|---|
| Источник 1 | **DailyMed** REST API (инструкции FDA, раздел 7 «Drug Interactions») | есть в реестре, бесплатно, без ключа |
| Источник 2 | **PubMed** через NCBI E-utilities | есть в реестре, бесплатно, ключ необязателен |
| LLM | **Groq**, `openai/gpt-oss-120b` (`reasoning_effort: low`); экономный вариант — `qwen/qwen3.8-27b`; запасной провайдер — OpenRouter `:free` | у автора нет платных ключей; `llama-3.3-70b-versatile` Groq убрал (404 `model_not_found`, ноябрь 2026) |
| Вызов LLM | OpenAI-совместимый `/chat/completions` через `requests`, без SDK | провайдер меняется одной строкой в секретах |
| UI / хостинг | Streamlit → Streamlit Community Cloud | проще всего для учебного проекта |

**Не использовать:** RxNav Interaction API (`rxnav.nlm.nih.gov/REST/interaction/...`) отключён
2 января 2024 года, эндпоинты больше не работают. Остальной RxNorm работает, его можно взять
как вспомогательный шаг нормализации названий (бренд → действующее вещество), но это не
третий «источник» без согласования с преподавателем. DrugBank не брать: сложная лицензия.
DDInter 2.0 есть только как идея на будущее (нет в реестре, нет документированного API).

## Архитектура

```
Streamlit (app.py)
   └─ orchestrator.run(drug_a, drug_b)
        ├─ dailymed_agent ─┐  параллельно (ThreadPoolExecutor)
        ├─ pubmed_agent  ──┘  → SourceReport (evidence + summary + errors)
        ├─ synthesizer   → ответ, каждое утверждение со ссылкой [DailyMed]/[PMID:…]
        └─ critic        → проверка: нет ли утверждений без опоры на evidence
```

Модели данных в `ddi/models.py`: `Evidence` → `SourceReport` → `FinalReport`.
Клиенты в `ddi/sources/` не вызывают LLM, это чистый HTTP и парсинг. Логика LLM живёт в `ddi/agents/`.

## Структура

```
app.py                     Streamlit UI (готов, рисует FinalReport)
ddi/config.py              Settings: st.secrets → env → .env
ddi/models.py              pydantic-модели
ddi/llm.py                 complete(system, user): groq|openrouter|gemini|ollama, ретраи на 429
ddi/sources/dailymed.py    готов: find_interactions(a, b) → (evidence, errors), поиск в обе стороны
ddi/sources/pubmed.py      TODO: search, fetch_abstracts
ddi/agents/retrievers.py   dailymed_agent подключён (без LLM-сводки); pubmed_agent — заглушка
ddi/agents/synthesizer.py  заглушка
ddi/agents/critic.py       заглушка
ddi/agents/orchestrator.py готов
tests/test_smoke.py        прогон пайплайна без сети
tests/test_dailymed.py     клиент DailyMed на урезанных реальных ответах (tests/fixtures/dailymed/)
```

## Статус

- [x] Каркас, конфиг, модели, UI, оркестратор
- [x] Обёртка LLM (`ddi/llm.py`), проверена реальным вызовом Groq
- [x] Клиент DailyMed (проверен вживую на 4 парах из шага 4)
- [ ] Клиент PubMed
- [ ] Агенты-поисковики с LLM-сводкой (DailyMed уже отдаёт evidence, сводки нет)
- [ ] Синтезатор и критик (промпты, формат ссылок)
- [ ] Кэширование (`st.cache_data`), обработка ошибок в UI
- [ ] Тесты клиентов (сохранённые ответы API, без сети): DailyMed готов, PubMed нет
- [ ] Деплой на Streamlit Community Cloud
- [ ] Презентация

## Особенности DailyMed (выяснено на живых данных)

- Без фильтра поиск возвращает в основном OTC-инструкции («Drug Facts»), где нет раздела
  Drug Interactions. Поэтому ищем только рецептурные: `doctype=34391-3`.
- LOINC `34073-7` ненадёжен: в старых инструкциях им бывает помечен посторонний раздел
  («Preexisting asthma»), а настоящий «Drug Interactions» помечен как unclassified.
  Раздел ищем по коду **или** по заголовку, «Drug/Laboratory Test Interactions» пропускаем.
- Название препарата часто есть только в заголовке подраздела («Warfarin-type anticoagulants»),
  поэтому фильтр смотрит и на заголовок, и на текст.
- Таблицы нового формата состоят из групп: строка-заголовок («Strong CYP3A4 inhibitors»), затем
  «Clinical Impact / Intervention / Examples». Группа идёт одним фрагментом.
- Таблицы бывают на тысячи символов: окно в 1 500 символов строится вокруг упоминания препарата.
- Классы и синонимы (NSAID, SSRI, strong CYP3A4 inhibitor…) лежат в `RELATED_TERMS`. Список
  неполный: остальные препараты ищутся только по названию (ограничение для доклада).
- У инструкций нового формата заголовок документа — дисклеймер «These highlights…», поэтому
  название берётся из результатов поиска.

## Следующие шаги (по порядку)

1. ~~DailyMed~~ — готово.
2. **PubMed.** `esearch.fcgi?db=pubmed&term=<a>[tiab] AND <b>[tiab] AND (drug interactions[mh] OR interaction[tiab])&retmax=10&retmode=json`,
   затем `efetch.fcgi?db=pubmed&id=...&retmode=xml` → заголовок, аннотация, год, PMID. Передавать
   `tool`, `email` и `api_key`, если они заданы. Лимит: около 3 запросов в секунду без ключа.
3. Агенты: retriever → краткая LLM-сводка по своим evidence. Синтезатор работает только по evidence
   и обязательно ставит ссылки. Критик возвращает список неподтверждённых утверждений.
4. Проверить вживую на парах `warfarin + ibuprofen`, `simvastatin + clarithromycin`,
   `sertraline + tramadol` и на паре без известного взаимодействия.
5. Деплой (см. README).

## Ограничения и правила

- **Лимиты Groq на бесплатном тарифе** (порядка 10 000 токенов в минуту, точные цифры по модели
  смотреть в console.groq.com). У `gpt-oss-120b` скрытые рассуждения тоже расходуют `max_tokens`. Обрезать evidence: топ-N фрагментов,
  примерно 1 500 символов на фрагмент. Не больше 4 вызовов LLM на один запрос пользователя.
- **Секреты:** только через `st.secrets` или `.env`. `.streamlit/secrets.toml` и `.env` в `.gitignore`,
  никогда не коммитить. Шаблон лежит в `.streamlit/secrets.toml.example`.
- Модель не должна выдумывать взаимодействия: если в evidence ничего нет, ответ должен быть
  «в источниках не найдено», а не «взаимодействия нет».
- В UI всегда показывать дисклеймер: справочный инструмент, не медицинская консультация.
- Названия препаратов вводятся на английском (DailyMed и PubMed англоязычные).
- Тесты не должны ходить в сеть: мокать `requests` или использовать фикстуры с сохранёнными ответами.
- Язык интерфейса и ответов русский, код и комментарии на английском.

## Команды

Локально (Windows): Python 3.12 через `py -3.12`, окружение в `.venv` (в `.gitignore`).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # вписать LLM_API_KEY (console.groq.com/keys)
.venv\Scripts\python -m streamlit run app.py
.venv\Scripts\python -m pytest
```

Репозиторий: https://github.com/teslicck/ddi_multiagent_system (ветка `main`).
