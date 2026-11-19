"""Sequential JSON-RPC exchanges over the three official SDK transports."""
import asyncio
import os
import sys
from pathlib import Path

import httpx2
from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCNotification, JSONRPCRequest, JSONRPCResponse

ROOT = Path(__file__).resolve().parents[1]
LOCAL_URL = "http://127.0.0.1:8000/sse"
PUBMED_URL = "https://pubmed.mcp.claude.com/mcp"


class ConnectionFailure(RuntimeError):
    pass


def transport(source, url):
    if source == "stdio":
        return stdio_client(StdioServerParameters(command=sys.executable,
                            args=["-m", "clinical_mcp.server"], cwd=ROOT,
                            env={"PYTHONUTF8": "1"}))
    if source == "sse":
        return sse_client(url or LOCAL_URL, timeout=2)
    if source == "remote":
        return streamable_http_client(url or os.getenv("PUBMED_MCP_URL", PUBMED_URL))
    raise ValueError(f"Неизвестный источник {source}")


def packet(message):
    return message.model_dump(mode="json", by_alias=True, exclude_unset=True)


async def exchange(reader, writer, request):
    await writer.send(SessionMessage(request))
    async for incoming in reader:
        if isinstance(incoming, Exception):
            raise incoming
        message = incoming.message
        if isinstance(message, JSONRPCRequest) and message.method == "ping":
            await writer.send(SessionMessage(JSONRPCResponse(jsonrpc="2.0", id=message.id, result={})))
        elif getattr(message, "id", None) == request.id:
            return {"request": packet(request), "response": packet(message)}
    raise ConnectionFailure("Сервер закрыл канал до получения ответа.")


async def session(source, method, params, url):
    async with transport(source, url) as (reader, writer):
        async def send(request_id, operation, arguments):
            request = JSONRPCRequest(jsonrpc="2.0", id=request_id, method=operation, params=arguments)
            return await exchange(reader, writer, request)

        init = await send(1, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                          "clientInfo": {"name": "ddi-agents", "version": "1.0"}})
        handshake = init["response"]
        if "error" in handshake:
            raise ConnectionFailure(handshake["error"]["message"])
        if handshake["result"].get("protocolVersion") not in {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}:
            raise ConnectionFailure("Сервер предложил неподдерживаемую версию протокола.")
        await writer.send(SessionMessage(JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")))
        reply = await send(2, method, params)
        if method == "tools/list":
            listing = reply["response"].get("result", {})
            tools = list(listing.get("tools", []))
            request_id = 2
            while cursor := listing.get("nextCursor"):
                request_id += 1
                page = await send(request_id, method, {"cursor": cursor})
                if "error" in page["response"]:
                    raise ConnectionFailure(page["response"]["error"]["message"])
                listing = page["response"]["result"]
                tools.extend(listing["tools"])
            reply["tools"] = tools
        return reply


def causes(exc):
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for child in exc.exceptions for leaf in causes(child)]
    return [exc]


async def execute(source, method, params=None, *, url=None, timeout=15):
    try:
        async with asyncio.timeout(timeout):
            for attempt in range(3):
                try:
                    return await session(source, method, params or {}, url)
                except Exception as exc:
                    errors = causes(exc)
                    if source == "sse" and attempt < 2 and all(isinstance(e, httpx2.ConnectError) for e in errors):
                        await asyncio.sleep(0.25 * (attempt + 1))
                        continue
                    if any(isinstance(e, httpx2.ConnectError) for e in errors):
                        raise ConnectionFailure("Не удалось установить сетевое соединение.") from exc
                    raise ConnectionFailure(str(errors[0]) or type(errors[0]).__name__) from exc
    except TimeoutError as exc:
        raise ConnectionFailure(f"Сервер не ответил за {timeout} с.") from exc


def sanitize(schema, values):
    """Use explicit schema defaults and normalize numeric text without replacing measurements."""
    clean = {}
    for name, field in schema.get("properties", {}).items():
        value = values.get(name)
        if value is None or value == "":
            if "default" in field:
                value = field["default"]
            elif name in schema.get("required", []):
                raise ValueError(f"Заполните поле «{field.get('title', name)}».")
            else:
                continue
        kind = field.get("type")
        if isinstance(value, str) and kind in {"number", "integer"}:
            try:
                value = (int if kind == "integer" else float)(value.strip().replace(",", "."))
            except ValueError as exc:
                raise ValueError(f"Проверьте число в поле «{field.get('title', name)}».") from exc
        clean[name] = value
    return clean
