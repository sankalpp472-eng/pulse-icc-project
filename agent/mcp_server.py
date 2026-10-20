"""
MCP servers for the tau-bench airline tools: one MCP server per tool.

Each tool is wrapped as its own MCP server (official MCP Python SDK, Streamable
HTTP transport, stateless + JSON responses so any replica can answer any call).

  TOOLS=all  python mcp_server.py   # local dev: every tool, each at /<tool>/mcp   (default)
  TOOLS=get_user_details python mcp_server.py
                                    # one tool at /mcp  -> one Knative service per tool

Env: TOOLS (comma list or "all"), PORT (default 9000; Knative sets 8080),
     REDIS_URL (optional shared database, e.g. redis://redis:6379/0).
Extra routes: GET /healthz (readiness), POST /reset (reload the airline database).

State:
  no REDIS_URL -> each process holds its own in-memory copy of the database. Fine when
                  one process serves all tools; on Knative every tool pod has its OWN copy.
  REDIS_URL    -> reservations and users live in Redis, shared by all tool pods, so a
                  write by one tool is seen by the others. Flights are never modified by
                  any tool, so each pod loads them once at start-up.
  python mcp_server.py --seed   loads the airline data into Redis (and a pristine copy
                                under seed:* that resets copy back between episodes)
"""

import json
import os
import sys
from collections.abc import MutableMapping
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from tau_airline import ALL_TOOLS, load_data

TOOLS = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
REDIS_URL = os.getenv("REDIS_URL")
TABLES = ("flights", "reservations", "users")


class RedisTable(MutableMapping):
    """One database table (reservations or users) stored as a Redis hash: key -> JSON.

    Records are fetched on first access and cached for the duration of ONE tool call;
    flush() writes back only the records that changed (or were added)."""

    def __init__(self, r, name: str):
        self.r, self.name, self.cache, self.orig = r, name, {}, {}

    def __getitem__(self, key):
        if key not in self.cache:
            raw = self.r.hget(self.name, key)
            if raw is None:
                raise KeyError(key)
            self.cache[key], self.orig[key] = json.loads(raw), raw
        return self.cache[key]

    def __contains__(self, key):
        return key in self.cache or bool(self.r.hexists(self.name, key))

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.orig.setdefault(key, None)

    def __delitem__(self, key):
        raise NotImplementedError("tools never delete records")

    def __iter__(self):
        return iter(self.r.hkeys(self.name))

    def __len__(self):
        return self.r.hlen(self.name)

    def flush(self):
        dirty = {k: s for k, v in self.cache.items() if (s := json.dumps(v)) != self.orig.get(k)}
        if dirty:
            self.r.hset(self.name, mapping=dirty)


def _redis():
    import redis
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def reset_redis(r) -> None:
    """Restore reservations and users from the pristine seed:* copies (explicit rewrite,
    so it never depends on how a Redis implementation does COPY)."""
    for t in ("reservations", "users"):
        pristine = r.hgetall(f"seed:{t}")
        pipe = r.pipeline()
        pipe.delete(t)
        pipe.hset(t, mapping=pristine)
        pipe.execute()


def seed_redis():
    """Load the airline data into Redis, plus a pristine copy (seed:<table>) used for resets."""
    r, data = _redis(), load_data()
    for t in TABLES:
        mapping = {k: json.dumps(v) for k, v in data[t].items()}
        for key in (t, f"seed:{t}"):
            r.delete(key)
            r.hset(key, mapping=mapping)
    print(f"seeded {REDIS_URL}: " + ", ".join(f"{t}={len(data[t])}" for t in TABLES))


if REDIS_URL and "--seed" not in sys.argv:
    _R = _redis()
    STATE = {"flights": {k: json.loads(v) for k, v in _R.hgetall("flights").items()}}
    if not STATE["flights"]:
        raise SystemExit(f"Redis at {REDIS_URL} has no airline data: run `python mcp_server.py --seed` first")
else:
    STATE = {"data": load_data()}


def invoke_tool(tool, args: dict) -> str:
    """Run a tau-bench tool on the in-memory database, or on the shared Redis one."""
    if not REDIS_URL:
        return tool.invoke(data=STATE["data"], **args)
    tables = {t: RedisTable(_R, t) for t in ("reservations", "users")}
    out = tool.invoke(data={"flights": STATE["flights"], **tables}, **args)
    for t in tables.values():
        t.flush()
    return out


def build_tool_server(name: str) -> Server:
    """An MCP server that exposes exactly one tau-bench tool, with its original schema."""
    tool = TOOLS[name]
    fn = tool.get_info()["function"]
    spec = types.Tool(name=fn["name"], description=fn["description"], input_schema=fn["parameters"])

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[spec])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name != name:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool {params.name}")], is_error=True)
        try:
            out, err = invoke_tool(tool, params.arguments or {}), False
        except Exception as e:  # bad arguments from the LLM -> report, don't crash
            out, err = f"{type(e).__name__}: {e}", True
        return types.CallToolResult(content=[types.TextContent(type="text", text=out)], is_error=err)

    return Server(f"pulse-{name}", version="0.1", on_list_tools=list_tools, on_call_tool=call_tool)


def build_app(names: list[str]) -> Starlette:
    # host="0.0.0.0" -> no localhost-only Host check, so Knative's Host header is accepted
    subs = {n: build_tool_server(n).streamable_http_app(json_response=True, stateless_http=True, host="0.0.0.0")
            for n in names}

    @asynccontextmanager
    async def lifespan(app):
        # Mounted sub-apps don't run their own lifespan; start every session manager here
        async with AsyncExitStack() as stack:
            for sub in subs.values():
                await stack.enter_async_context(sub.router.lifespan_context(sub))
            yield

    async def healthz(request):
        return PlainTextResponse("ok")

    async def reset(request):
        if REDIS_URL:
            reset_redis(_R)
        else:
            STATE["data"] = load_data()
        return PlainTextResponse("database reloaded")

    routes = [Route("/healthz", healthz), Route("/reset", reset, methods=["POST"])]
    routes += [Mount(f"/{n}", app=sub) for n, sub in subs.items()]   # /<tool>/mcp
    if len(subs) == 1:
        routes.append(Mount("/", app=next(iter(subs.values()))))    # /mcp for a single-tool service
    return Starlette(routes=routes, lifespan=lifespan)


def main():
    if "--seed" in sys.argv:
        if not REDIS_URL:
            raise SystemExit("--seed needs REDIS_URL")
        return seed_redis()
    sel = os.getenv("TOOLS", "all")
    names = list(TOOLS) if sel == "all" else [n.strip() for n in sel.split(",") if n.strip()]
    unknown = set(names) - set(TOOLS)
    if unknown:
        raise SystemExit(f"Unknown tools: {sorted(unknown)}. Available: {sorted(TOOLS)}")
    port = int(os.getenv("PORT", "9000"))
    print(f"Database: {'Redis ' + REDIS_URL if REDIS_URL else 'in-memory'}")
    print(f"Serving {len(names)} MCP tool server(s) on :{port}: "
          + ", ".join(f"/{n}/mcp" for n in names) + (" (+ /mcp)" if len(names) == 1 else ""))
    uvicorn.run(build_app(names), host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
