"""Use the MCP server as the agent's tool backend.

The agent is synchronous (simpler to read and debug), while the MCP client is async.
We run one long-lived async task in a background thread that owns the MCP session,
and send it tool calls through a queue. Same interface as LocalToolBackend.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.config import ROOT


class McpToolBackend:
    def __init__(self, command: list[str] | None = None, timeout: float = 20.0, env: dict[str, str] | None = None):
        cmd = command or [sys.executable, "-m", "mcp_server.menu_server"]
        self._params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(ROOT), env=env)
        self._timeout = timeout
        self._schemas: list[dict[str, Any]] = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._queue: asyncio.Queue | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("MCP server did not start in time")
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error}")

    # ------------------------------------------------ background thread
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._queue = asyncio.Queue()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._schemas = [
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema", {}),  # SDK 2.x | 1.x
                            },
                        }
                        for t in listed.tools
                    ]
                    self._ready.set()
                    while True:
                        job = await self._queue.get()
                        if job is None:
                            break
                        name, args, fut = job
                        try:
                            res = await session.call_tool(name, args)
                            text = "".join(getattr(c, "text", "") for c in res.content)
                            fut.set_result(text)
                        except Exception as e:  # noqa: BLE001
                            fut.set_exception(e)
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._ready.set()

    # ------------------------------------------------ public API
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        fut: concurrent.futures.Future[str] = concurrent.futures.Future()
        clean = {k: v for k, v in (args or {}).items() if v is not None}
        assert self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, clean, fut))
        return fut.result(timeout=self._timeout)

    def close(self) -> None:
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        self._thread.join(timeout=5)
