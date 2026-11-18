"""
MCP servers for the tau-bench tools (airline or retail): one MCP server per tool.

Each tool is wrapped as its own MCP server (official MCP Python SDK, Streamable
HTTP transport, stateless + JSON responses so any replica can answer any call).

  TOOLS=all  python mcp_server.py   # local dev: every tool, each at /<tool>/mcp   (default)
  TOOLS=get_user_details python mcp_server.py
                                    # one tool at /mcp  -> one Knative service per tool
  DOMAIN=retail TOOLS=all python mcp_server.py      # the 16 retail tools instead

Env: DOMAIN (airline | retail, default airline), TOOLS (comma list or "all"),
     PORT (default 9000; Knative sets 8080),
     REDIS_URL (optional shared database, e.g. redis://redis:6379/0; retail uses its own
     Redis DB, e.g. .../1, because both domains have a "users" table).
Extra routes: GET /healthz (readiness), POST /reset (reload the domain's database).

State:
  no REDIS_URL -> each process holds its own in-memory copy of the database. Fine when
                  one process serves all tools; on Knative every tool pod has its OWN copy.
  REDIS_URL    -> the tables tools write (airline: reservations, users; retail: orders,
                  users) live in Redis, shared by all tool pods, so a write by one tool is
                  seen by the others. The read-only table (airline: flights; retail:
                  products) is loaded once per pod at start-up.
  python mcp_server.py --seed   loads the domain's data into Redis (and a pristine copy
                                under seed:* that resets copy back between episodes)
  Redis keeps no data across a restart; a tool that finds it empty reseeds it from the data in
  the image (one pod at a time, under seed:lock). While Redis is down a pod still starts and
  its calls fail within seconds with "DatabaseUnavailable: ..." instead of crash-looping.
"""

import json
import os
import random
import sys
import time
from collections.abc import MutableMapping
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

import pulse.agent.tau_airline as tau_airline
import pulse.agent.tau_retail as tau_retail

# domain -> (package, read-only table, tables the tools write)
DOMAINS = {"airline": (tau_airline, "flights", ("reservations", "users")),
           "retail": (tau_retail, "products", ("orders", "users"))}
DOMAIN = os.getenv("DOMAIN", "airline")
if DOMAIN not in DOMAINS:
    raise SystemExit(f"Unknown DOMAIN={DOMAIN!r}; use one of {sorted(DOMAINS)}")
_PKG, STATIC, SHARED = DOMAINS[DOMAIN]
load_data = _PKG.load_data

TOOLS = {t.get_info()["function"]["name"]: t for t in _PKG.ALL_TOOLS}
REDIS_URL = os.getenv("REDIS_URL")
TABLES = (STATIC, *SHARED)


class RedisTable(MutableMapping):
    """One database table (e.g. reservations or users) stored as a Redis hash: key -> JSON.

    Records are fetched on first access and cached for the duration of ONE tool call;
    commit() writes back only the records that changed (or were added). Iterating loads the
    whole table in one HGETALL (retail's find_user_id_* scan every user)."""

    def __init__(self, r, name: str):
        self.r, self.name, self.cache, self.orig = r, name, {}, {}
        self.loaded_all = False

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
        if not self.loaded_all:
            for k, raw in self.r.hgetall(self.name).items():
                if k not in self.cache:
                    self.cache[k], self.orig[k] = json.loads(raw), raw
            self.loaded_all = True
        return iter(list(self.cache))

    def __len__(self):
        return self.r.hlen(self.name)

    def dirty(self) -> dict:
        """The records this call changed or added, as JSON."""
        return {k: s for k, v in self.cache.items() if (s := json.dumps(v)) != self.orig.get(k)}


class WriteConflict(RuntimeError):
    """Another tool call changed a record this call read, between our read and our write (issue #31)."""


def commit(r, tables: list[RedisTable]) -> None:
    """Write back what one tool call changed, atomically and only if nothing it read changed meanwhile.

    A tool call is a read-modify-write of whole records (e.g. book_reservation reads the user, takes the payment,
    writes the user back). Two concurrent calls on different pods used to both read the same version and the later
    HSET silently overwrote the earlier one (a lost update). Now: WATCH the tables, check that every record this call
    read still has the value it read (a record it created must still be absent), then MULTI/HSET/EXEC. If anything
    changed, raise WriteConflict and invoke_tool reruns the call on fresh data. Read-only calls write nothing and skip
    this entirely."""
    dirty = {t.name: d for t in tables if (d := t.dirty())}
    if not dirty:
        return
    import redis
    with r.pipeline() as pipe:
        try:
            pipe.watch(*(t.name for t in tables))
            for t in tables:
                keys = list(t.orig)
                if keys and pipe.hmget(t.name, keys) != [t.orig[k] for k in keys]:
                    raise WriteConflict(f"{t.name} changed during the call")
            pipe.multi()
            for name, mapping in dirty.items():
                pipe.hset(name, mapping=mapping)
            pipe.execute()
        except redis.exceptions.WatchError as e:
            raise WriteConflict("a watched table changed before the write") from e


WRITE_RETRIES = int(os.getenv("TOOL_WRITE_RETRIES", "8"))   # reruns of a tool call that lost a write race
REDIS_TIMEOUT_S = float(os.getenv("REDIS_TIMEOUT_S", "3"))   # connect / command timeout: fail fast, never hang


class DatabaseUnavailable(RuntimeError):
    """The shared Redis database cannot be reached (or is being reseeded): the tool call fails with this
    message instead of hanging or crashing the pod."""


def _redis():
    import redis
    from redis.backoff import ExponentialBackoff
    from redis.retry import Retry
    # A Redis restart breaks the pooled connections: retry a command twice on a fresh one before giving up
    return redis.Redis.from_url(REDIS_URL, decode_responses=True,
                                socket_connect_timeout=REDIS_TIMEOUT_S, socket_timeout=REDIS_TIMEOUT_S,
                                retry=Retry(ExponentialBackoff(cap=0.5, base=0.05), 2),
                                retry_on_error=[redis.exceptions.ConnectionError, redis.exceptions.TimeoutError])


def _redis_errors() -> tuple[type[BaseException], ...]:
    try:
        from redis.exceptions import RedisError
        return (RedisError,)
    except ImportError:
        return ()


def write_seed(r, data: dict) -> None:
    """Load every table and its pristine seed:<table> copy; one transaction per table, so a reader never sees a
    half-written table and two pods seeding at once leave the same result."""
    for t in TABLES:
        mapping = {k: json.dumps(v) for k, v in data[t].items()}
        pipe = r.pipeline(transaction=True)
        for key in (t, f"seed:{t}"):
            pipe.delete(key)
            pipe.hset(key, mapping=mapping)
        pipe.execute()


SEED_LOCK = "seed:lock"


def ensure_seeded(r, wait_s: float = 20.0) -> bool:
    """Make sure the domain's data is in Redis. Redis keeps no data across a restart (redis.yaml), so after one
    the first tool that notices reloads the data baked into this image; the other pods wait for it.
    Returns True if this call (re)seeded."""
    keys = [*TABLES, *(f"seed:{t}" for t in TABLES)]
    if r.exists(*keys) == len(keys):
        return False
    if r.set(SEED_LOCK, os.getenv("HOSTNAME", "tool"), nx=True, ex=60):
        try:
            if r.exists(*keys) != len(keys):
                print(f"Redis at {REDIS_URL} has no (or partial) {DOMAIN} data (restarted?): reseeding it",
                      flush=True)
                write_seed(r, load_data())
                return True
            return False
        finally:
            r.delete(SEED_LOCK)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:                # another pod is seeding
        time.sleep(0.2)
        if r.exists(*keys) == len(keys):
            return False
    raise DatabaseUnavailable(f"the shared database (Redis at {REDIS_URL}) is being reseeded: try again")


def reset_redis(r) -> None:
    """Restore the written tables from the pristine seed:* copies (explicit rewrite,
    so it never depends on how a Redis implementation does COPY)."""
    if ensure_seeded(r):                                 # just reloaded from scratch = already pristine
        return
    for t in SHARED:
        pristine = r.hgetall(f"seed:{t}")
        pipe = r.pipeline()
        pipe.delete(t)
        pipe.hset(t, mapping=pristine)
        pipe.execute()


def seed_redis():
    """Load the domain's data into Redis, plus a pristine copy (seed:<table>) used for resets."""
    r, data = _redis(), load_data()
    write_seed(r, data)
    print(f"seeded {DOMAIN} data into {REDIS_URL}: " + ", ".join(f"{t}={len(data[t])}" for t in TABLES))


_R = None


def shared_db():
    """The Redis client, with the domain's data present and the read-only table loaded in this pod.
    Raises DatabaseUnavailable (naming the cause) if Redis is down; the next call tries again."""
    global _R
    try:
        if _R is None:
            _R = _redis()
        ensure_seeded(_R)
        if not STATE.get(STATIC):
            STATE[STATIC] = {k: json.loads(v) for k, v in _R.hgetall(STATIC).items()}
    except _redis_errors() as e:
        raise DatabaseUnavailable(f"the shared database (Redis at {REDIS_URL}) is unavailable: "
                                  f"{type(e).__name__}: {e}") from e
    return _R


if REDIS_URL and "--seed" not in sys.argv:
    # Load the read-only table now so a warm pod answers at once. If Redis is down the pod still starts:
    # its calls fail fast with the cause, and recover by themselves once Redis is back.
    STATE = {}
    try:
        shared_db()
    except DatabaseUnavailable as e:
        print(f"WARNING: {e} (tool calls will fail until it is back)", flush=True)
else:
    STATE = {"data": load_data()}


def invoke_tool(tool, args: dict) -> str:
    """Run a tau-bench tool on the in-memory database, or on the shared Redis one."""
    if not REDIS_URL:
        return tool.invoke(data=STATE["data"], **args)
    r = shared_db()
    for attempt in range(WRITE_RETRIES + 1):
        tables = {t: RedisTable(r, t) for t in SHARED}       # fresh reads on every attempt
        try:
            out = tool.invoke(data={STATIC: STATE[STATIC], **tables}, **args)
            commit(r, list(tables.values()))
            return out
        except WriteConflict:
            if attempt == WRITE_RETRIES:
                raise WriteConflict(f"gave up after {WRITE_RETRIES + 1} attempts: other calls kept changing the "
                                    f"same records") from None
            time.sleep(random.uniform(0, 0.005 * 2 ** attempt))   # back off a little, then rerun on fresh data
        except _redis_errors() as e:
            raise DatabaseUnavailable(f"the shared database (Redis at {REDIS_URL}) failed during the call: "
                                      f"{type(e).__name__}: {e}") from e


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
        except DatabaseUnavailable as e:  # Redis down -> say so, fast
            out, err = f"DatabaseUnavailable: {e}", True
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
            try:
                reset_redis(shared_db())
            except (DatabaseUnavailable, *_redis_errors()) as e:
                return PlainTextResponse(f"reset failed: {e}", status_code=503)
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
    print(f"Domain: {DOMAIN}. Database: {'Redis ' + REDIS_URL if REDIS_URL else 'in-memory'}")
    print(f"Serving {len(names)} MCP tool server(s) on :{port}: "
          + ", ".join(f"/{n}/mcp" for n in names) + (" (+ /mcp)" if len(names) == 1 else ""))
    uvicorn.run(build_app(names), host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
