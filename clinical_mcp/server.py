"""Run with python -m clinical_mcp.server [--sse]."""
import argparse
import logging
import sys

from mcp.server import MCPServer

from .calculators import NETWORK_TOOLS, STDIO_TOOLS


def build_server(network=False):
    server = MCPServer("Клинические инструменты")
    for tool in NETWORK_TOOLS if network else STDIO_TOOLS:
        server.add_tool(tool)

    @server.tool()
    async def list_calculators() -> dict:
        """Справочник методик, единиц измерения и допустимых значений."""
        return {tool.name: {"description": tool.description, "parameters": tool.input_schema}
                for tool in await server.list_tools() if tool.name != "list_calculators"}

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = build_server(args.sse)
    if args.sse:
        server.run(transport="sse", host="127.0.0.1", port=args.port)
    else:
        server.run()
