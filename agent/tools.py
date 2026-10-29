"""
Tool backends for the agent. Both have the same interface (`schemas`, `call`),
so the agent loop does not change:

  LocalToolBackend -- tools run in-process on a fresh copy of the airline database
  MCPToolBackend   -- each call goes to that tool's own MCP server (mcp_server.py),
                      locally or as a serverless function on Knative

Pick with TOOL_BACKEND=local|mcp (run_tasks.py --tools does this for you).
DOMAIN=airline|retail picks the tau-bench domain (default airline). For the retail tools on Knative
use MCP_HOST_TEMPLATE=retail-{svc}.default.example.com (deploy/knative/gen_services.py --domain retail).
"""

import asyncio
import hashlib
import json
import os
import time

DOMAIN = os.getenv("DOMAIN", "airline")
if DOMAIN == "retail":
    from tau_retail import ALL_TOOLS, load_data
else:
    from tau_airline import ALL_TOOLS, load_data

# Tools that end the conversation when called (same as tau-bench)
TERMINATE_TOOLS = {"transfer_to_human_agents"}
# Tools that only read the database: several calls of these can run at once
READ_ONLY_TOOLS = {
    "airline": {"get_user_details", "get_reservation_details", "search_direct_flight", "search_onestop_flight",
                "list_all_airports", "calculate"},
    "retail": {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
               "get_product_details", "list_all_product_types", "calculate"},
}[DOMAIN if DOMAIN == "retail" else "airline"]

# Every MCP call, appended by MCPToolBackend: experiment runners read this to time calls
# (absolute start time, latency) even when an episode stops midway. Clear it between episodes.
CALL_LOG: list[dict] = []


class LocalToolBackend:
    def __init__(self):
        self.data = load_data()  # each episode gets its own fresh DB copy
        self._tools = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
        self.schemas = [t.get_info() for t in ALL_TOOLS]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> tuple[str, float]:
        """Run a tool. Returns (observation, latency_seconds)."""
        t0 = time.perf_counter()
        if name not in self._tools:
            out = f"Error: unknown tool {name}"
        else:
            try:
                out = self._tools[name].invoke(data=self.data, **args)
            except Exception as e:  # bad args from the LLM -> tell it, don't crash
                out = f"Error: {type(e).__name__}: {e}"
        return out, time.perf_counter() - t0

    def data_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()


class MCPToolBackend:
    """Calls each tool on its own MCP server over Streamable HTTP.

    URL per tool:  MCP_URL_TEMPLATE  (default http://127.0.0.1:9000/{tool}/mcp, i.e. mcp_server.py locally)
    Host header:   MCP_HOST_TEMPLATE (optional, e.g. "{svc}.default.example.com" for Knative behind Kourier)
    Placeholders:  {tool} = tool name (get_user_details), {svc} = DNS-safe name (get-user-details)

    Tool schemas are static (the same tau-bench definitions the servers expose).
    We never call list_tools up front: that would contact -- and wake -- every tool.
    The database lives in the servers, so there is no local data_hash (no scoring).
    """

    def __init__(self):
        # Import the MCP client here, not on the first call: the ~1 s import would
        # otherwise be added to the first tool call and look like a cold start.
        from mcp import ClientSession
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        self._ClientSession, self._make_http, self._transport = \
            ClientSession, create_mcp_http_client, streamable_http_client

        self.schemas = [t.get_info() for t in ALL_TOOLS]
        self._names = {s["function"]["name"] for s in self.schemas}
        self.url_tpl = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:9000/{tool}/mcp")
        self.host_tpl = os.getenv("MCP_HOST_TEMPLATE") or None

    @property
    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self.schemas]

    def call(self, name: str, args: dict) -> tuple[str, float]:
        """Run a tool on its MCP server. Returns (observation, latency_seconds)."""
        out, latency, _ = asyncio.run(self.acall(name, args))
        return out, latency

    async def acall(self, name: str, args: dict) -> tuple[str, float, float]:
        """Async version of call, so several calls can run at once (asyncio.gather).
        Returns (observation, latency_seconds, wall-clock start)."""
        t_wall, t0 = time.time(), time.perf_counter()
        if name not in self._names:
            return f"Error: unknown tool {name}", 0.0, t_wall
        transport_ok = True
        try:
            out = await self._call(name, args)
        except Exception as e:  # server down, timeout ... -> the agent sees an error, run continues
            out, transport_ok = f"Error: MCP call to {name} failed: {type(e).__name__}: {e}", False
        latency = time.perf_counter() - t0
        CALL_LOG.append({"tool": name, "t_start": t_wall, "latency_s": latency, "transport_ok": transport_ok,
                         "tool_error": out.startswith("Error"), "result_head": out[:80]})
        return out, latency, t_wall

    async def _call(self, name: str, args: dict) -> str:
        names = {"tool": name, "svc": name.replace("_", "-")}
        headers = {"Host": self.host_tpl.format(**names)} if self.host_tpl else None
        async with self._make_http(headers=headers) as http, \
                self._transport(self.url_tpl.format(**names), http_client=http,
                                terminate_on_close=False) as (read, write), \
                self._ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(name, args)
        text = "".join(c.text for c in res.content if getattr(c, "type", None) == "text")
        return f"Error: {text}" if res.is_error else text


def make_tool_backend():
    kind = os.getenv("TOOL_BACKEND", "local").lower()
    if kind == "mcp":
        return MCPToolBackend()
    if kind == "local":
        return LocalToolBackend()
    raise ValueError(f"Unknown TOOL_BACKEND: {kind!r}. Use 'local' or 'mcp'.")
