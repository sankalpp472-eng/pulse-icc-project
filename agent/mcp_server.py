"""
MCP servers for the tau-bench airline tools: one MCP server per tool.

Each tool is wrapped as its own MCP server (official MCP Python SDK, Streamable
HTTP transport, stateless + JSON responses so any replica can answer any call).

  TOOLS=all  python mcp_server.py   # local dev: every tool, each at /<tool>/mcp   (default)
  TOOLS=get_user_details python mcp_server.py
                                    # one tool at /mcp  -> one Knative service per tool

Env: TOOLS (comma list or "all"), PORT (default 9000; Knative sets 8080).
Extra routes: GET /healthz (readiness), POST /reset (reload the airline database).

Note on state: all tools served by ONE process share one in-memory database,
so writes are visible to later reads. Separate processes (one per tool, e.g. on
Knative) each hold their own copy until the database moves to a shared store.
"""

import os
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from tau_airline import ALL_TOOLS, load_data

STATE = {"data": load_data()}
TOOLS = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}


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
            out, err = tool.invoke(data=STATE["data"], **(params.arguments or {})), False
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
        STATE["data"] = load_data()
        return PlainTextResponse("database reloaded")

    routes = [Route("/healthz", healthz), Route("/reset", reset, methods=["POST"])]
    routes += [Mount(f"/{n}", app=sub) for n, sub in subs.items()]   # /<tool>/mcp
    if len(subs) == 1:
        routes.append(Mount("/", app=next(iter(subs.values()))))    # /mcp for a single-tool service
    return Starlette(routes=routes, lifespan=lifespan)


def main():
    sel = os.getenv("TOOLS", "all")
    names = list(TOOLS) if sel == "all" else [n.strip() for n in sel.split(",") if n.strip()]
    unknown = set(names) - set(TOOLS)
    if unknown:
        raise SystemExit(f"Unknown tools: {sorted(unknown)}. Available: {sorted(TOOLS)}")
    port = int(os.getenv("PORT", "9000"))
    print(f"Serving {len(names)} MCP tool server(s) on :{port}: "
          + ", ".join(f"/{n}/mcp" for n in names) + (" (+ /mcp)" if len(names) == 1 else ""))
    uvicorn.run(build_app(names), host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
