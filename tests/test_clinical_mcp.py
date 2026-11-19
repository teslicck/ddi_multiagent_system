"""Exercise the page and actual JSON-RPC transports without an external account."""
import asyncio
from contextlib import contextmanager
import json
import socket
from threading import Thread
import time

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from clinical_mcp.client import ConnectionFailure, ROOT, execute, sanitize
from clinical_mcp.calculators import cockcroft_gault, glasgow, mosteller
from clinical_mcp.server import build_server


@contextmanager
def serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=8)


def call(source, name, args, **kwargs):
    reply = asyncio.run(execute(source, "tools/call", {"name": name, "arguments": args}, **kwargs))
    assert reply["request"]["id"] == reply["response"]["id"]
    return reply["response"]["result"]


def test_stdio_page():
    page = next((ROOT / "pages").glob("*MCP.py"))
    ui = AppTest.from_file(str(page), default_timeout=45).run()
    assert not ui.exception and not ui.error
    ui.toggle[0].set_value(True).run()
    ui.button[0].click().run()
    assert not ui.exception
    assert ui.metric[0].value == "22.86 кг/м²"
    assert ui.expander[0].label == "Инспектор JSON-RPC"


@pytest.mark.parametrize("source", ["sse", "remote"])
def test_network_transports(source):
    server = build_server(network=True)
    app = server.sse_app() if source == "sse" else server.streamable_http_app()
    with serve(app) as base:
        url = base + ("/sse" if source == "sse" else "/mcp")
        listing = asyncio.run(execute(source, "tools/list", url=url))
        assert len(listing["tools"]) == 6
        args = {"systolic": 100, "respirations": 22, "altered_mentation": False}
        body = call(source, "qsofa", args, url=url)
        assert json.loads(body["content"][0]["text"])["value"] == 2
        bad = call(source, "qsofa", dict(args, systolic=-1), url=url)
        assert bad["isError"] and "systolic" in bad["content"][0]["text"]
        catalog = call(source, "list_calculators", {}, url=url)
        assert "minimum" in catalog["content"][0]["text"]


def test_unavailable_port():
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{reserved.getsockname()[1]}/sse"
        with pytest.raises(ConnectionFailure):
            asyncio.run(execute("sse", "tools/list", url=url))


def test_timeout(monkeypatch):
    async def stalled(*args):
        await asyncio.sleep(10)
    monkeypatch.setattr("clinical_mcp.client.session", stalled)
    with pytest.raises(ConnectionFailure, match="не ответил"):
        asyncio.run(execute("remote", "tools/list", timeout=0.02))


def test_sanitizer():
    schema = {"properties": {"weight": {"type": "number"}, "limit": {"type": "integer", "default": 3}},
              "required": ["weight"]}
    assert sanitize(schema, {"weight": "70,5"}) == {"weight": 70.5, "limit": 3}
    with pytest.raises(ValueError, match="weight"):
        sanitize(schema, {})


def test_calculator_reference_values():
    assert cockcroft_gault(68, 72, 1, "мужской")["value"] == 72
    assert cockcroft_gault(68, 72, 1, "женский")["value"] == 61.2
    assert glasgow(1, 1, 1)["value"] == 3
    assert glasgow(4, 5, 6)["value"] == 15
    assert mosteller(80, 180)["value"] == 2
