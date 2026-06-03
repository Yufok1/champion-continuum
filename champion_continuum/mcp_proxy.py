from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
try:  # modern MCP servers speak streamable-HTTP; older libs may lack the client
    from mcp.client.streamable_http import streamablehttp_client
except Exception:  # pragma: no cover
    streamablehttp_client = None

logger = logging.getLogger("champion_continuum.mcp")


async def _open_url(stack, url: str):
    """Open an HTTP MCP transport, choosing by URL: '/sse' -> SSE, else streamable-HTTP.
    Returns (read, write). streamable-HTTP yields a third session-id value we drop."""
    parsed_path = urlparse(url).path.rstrip("/")
    if parsed_path.endswith("/sse") or streamablehttp_client is None:
        read, write = await stack.enter_async_context(sse_client(url))
        return read, write
    read, write, _ = await stack.enter_async_context(streamablehttp_client(url))
    return read, write


class MCPProxy:
    """Synchronous proxy for Model Context Protocol (MCP) clients."""

    def __init__(self, root: str | Path = ".continuum") -> None:
        self.root = Path(root)
        self.config_path = self.root / "mcp.json"

    def load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to load MCP config: {exc}")
            return {}

    async def list_tools(self) -> list[dict[str, Any]]:
        all_tools = []
        config = self.load_config()
        servers = config.get("mcpServers", {})
        if not servers:
            return []

        async with contextlib.AsyncExitStack() as stack:
            for name, params in servers.items():
                try:
                    if "url" in params:
                        read, write = await _open_url(stack, params["url"])
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        result = await session.list_tools()
                        for tool in result.tools:
                            all_tools.append({
                                "server": name,
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            })
                    elif "command" in params:
                        stdio_params = StdioServerParameters(
                            command=params["command"],
                            args=params.get("args", []),
                            env=params.get("env"),
                        )
                        read, write = await stack.enter_async_context(stdio_client(stdio_params))
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        result = await session.list_tools()
                        for tool in result.tools:
                            all_tools.append({
                                "server": name,
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            })
                except Exception as exc:
                    logger.error(f"Failed to list tools for {name}: {exc}")
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        config = self.load_config()
        servers = config.get("mcpServers", {})
        params = servers.get(server_name) if server_name else None
        if params is None and len(servers) == 1:
            # Forgiving routing: one server connected -> use it regardless of the
            # prefix the model guessed (e.g. 'mcp.get_help' -> the sole server).
            server_name, params = next(iter(servers.items()))
        if not params:
            raise ValueError(f"Unknown MCP server: {server_name}")

        async with contextlib.AsyncExitStack() as stack:
            if "url" in params:
                read, write = await _open_url(stack, params["url"])
            elif "command" in params:
                stdio_params = StdioServerParameters(
                    command=params["command"],
                    args=params.get("args", []),
                    env=params.get("env"),
                )
                read, write = await stack.enter_async_context(stdio_client(stdio_params))
            else:
                raise ValueError(f"Invalid config for server: {server_name}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result

    def get_tools_sync(self) -> list[dict[str, Any]]:
        """Sync wrapper for listing tools."""
        try:
            return anyio.run(self.list_tools)
        except Exception as exc:
            logger.error(f"Sync list_tools failed: {exc}")
            return []

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Sync wrapper for calling a tool."""
        return anyio.run(self.call_tool, server_name, tool_name, arguments)
