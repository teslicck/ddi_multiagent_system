import asyncio
import json
import os
import time

import streamlit as st

from clinical_mcp.client import ConnectionFailure, PUBMED_URL, execute, sanitize

st.set_page_config(page_title="Клинические инструменты MCP", layout="centered")
st.title("Клинические инструменты MCP")
SOURCES = {"stdio": "Локальный stdio", "sse": "Локальный SSE · порт 8000", "remote": "Внешний PubMed"}
EXAMPLE = dict(weight=70.0, height=175.0, age=60, creatinine=1.0, sex="мужской",
               eyes=4, verbal=5, motor=6, respirations=22, systolic=100, altered_mentation=False)


@st.cache_data(ttl=300, show_spinner=False)
def discover(source, endpoint):
    reply = asyncio.run(execute(source, "tools/list", url=endpoint))
    if error := reply["response"].get("error"):
        raise ConnectionFailure(error["message"])
    return reply["tools"]


def input_field(name, field, example):
    label = field.get("title", name)
    default = EXAMPLE.get(name, field.get("default")) if example else field.get("default")
    options = {"help": field.get("description"), "key": f"{source}:{tool['name']}:{name}:{example}"}
    if "anyOf" in field:
        variants = [v for v in field["anyOf"] if v.get("type") != "null"]
        if len(variants) == 1:
            field = {**field, **variants[0]}
    kind = field.get("type")
    if "enum" in field:
        choices = field["enum"]
        return st.selectbox(label, choices, index=choices.index(default) if default in choices else None, **options)
    if kind == "boolean":
        return st.checkbox(label, value=bool(default), **options)
    if kind in {"number", "integer"}:
        cast = int if kind == "integer" else float
        return st.number_input(label, value=None if default is None else cast(default),
                               min_value=cast(field["minimum"]) if "minimum" in field else None,
                               max_value=cast(field["maximum"]) if "maximum" in field else None,
                               step=1 if kind == "integer" else 0.01, **options)
    if kind in {"array", "object"} or "$ref" in field or ("anyOf" in field and not kind):
        raw = st.text_area(f"{label} (JSON)", value=json.dumps(default) if default is not None else "", **options)
        return json.loads(raw) if raw.strip() else None
    return st.text_input(label, value=default or "", **options)


def show_reply(reply):
    response = reply["response"]
    body = response.get("result", {})
    data = body.get("structuredContent") or {}
    if source != "remote" and not data and not body.get("isError") and body.get("content"):
        data = json.loads(body["content"][0]["text"])
    if "error" in response:
        st.warning(response["error"]["message"])
    elif body.get("isError"):
        for item in body.get("content", []):
            st.warning(item.get("text", "Ошибка вызова инструмента"))
    elif source != "remote" and "components" in data:
        st.metric("Результат", f"{data['value']} {data['unit']}")
        getattr(st, data["color"])(data["interpretation"])
        st.table([{"Компонент": k, "Значение": str(v)} for k, v in data["components"].items()])
        st.write(data["advice"])
        st.link_button("Источник методики", data["reference"])
    elif source == "remote" and isinstance(data.get("pmids"), list):
        st.metric("Публикаций по запросу", data.get("total_count", len(data["pmids"])))
        for pmid in data["pmids"]:
            if str(pmid).isdigit():
                st.markdown(f"[PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    elif data:
        st.json(data)
    else:
        for item in body.get("content", []):
            value = item.get("text")
            if value:
                try:
                    st.json(json.loads(value))
                except ValueError:
                    st.text(value)
            else:
                st.json(item)
    with st.expander("Инспектор JSON-RPC"):
        st.caption("Запрос")
        st.json(reply["request"])
        st.caption("Ответ")
        st.json(response)


source = st.radio("Источник MCP", list(SOURCES), format_func=SOURCES.get, horizontal=True)
endpoint = os.getenv("PUBMED_MCP_URL", PUBMED_URL) if source == "remote" else None
try:
    with st.spinner("Получаем список инструментов…"):
        tools = discover(source, endpoint)
except ConnectionFailure as exc:
    st.error(f"Не удалось подключиться к источнику «{SOURCES[source]}». {exc}")
    if source == "sse":
        st.code("python -m clinical_mcp.server --sse")
    st.button("Повторить подключение")
    st.stop()

if not tools:
    st.info("Сервер вернул пустой список инструментов.")
    st.stop()


def tool_label(item):
    if item["name"] == "search_articles":
        return "Поиск публикаций PubMed · search_articles"
    return (item.get("title") or item.get("description") or item["name"]).splitlines()[0] + f" · {item['name']}"


tool = st.selectbox("Найдите инструмент по названию", tools, format_func=tool_label)
schema = tool["inputSchema"]
example = source != "remote" and tool["name"] != "list_calculators" and st.toggle("Заполнить учебный пример")
if source == "remote":
    st.caption("Введите поисковую фразу по теме исследования или названия препаратов.")
with st.form(f"params:{source}:{tool['name']}"):
    try:
        values = {name: input_field(name, field, example) for name, field in schema.get("properties", {}).items()}
        input_error = None
    except ValueError as exc:
        input_error = str(exc)
    submitted = st.form_submit_button("Вызвать инструмент")

if submitted:
    try:
        if input_error:
            raise ValueError(input_error)
        arguments = sanitize(schema, values)
        params = {"name": tool["name"], "arguments": arguments}
        key = json.dumps([endpoint, params], sort_keys=True)
        cache = st.session_state.setdefault("pubmed_mcp_results", {})
        now = time.monotonic()
        for expired in [k for k, (until, _) in cache.items() if until <= now]:
            del cache[expired]
        if source == "remote" and key in cache:
            st.caption("Из кэша этой сессии. Инспектор показывает исходный обмен.")
            reply = cache[key][1]
        else:
            with st.spinner("Ожидаем ответ MCP…"):
                reply = asyncio.run(execute(source, "tools/call", params, url=endpoint))
            body = reply["response"].get("result")
            if source == "remote" and body and not body.get("isError"):
                cache[key] = (now + 900, reply)
        show_reply(reply)
    except (ValueError, ConnectionFailure) as exc:
        st.warning(f"Вызов не выполнен. {exc}")
