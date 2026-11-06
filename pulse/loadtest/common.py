"""
Shared by the load tests (pulse/loadtest/*.py):

  episodes()        the recorded W1 episodes from the replay's plan() (pulse/experiments/replay.py): same
                    seed = same traces, arguments, think / customer gaps, fan-out groups and fair hints
                    (with the signals and the conversation history, when the tau-bench data is there)
  hint_stream()     an endless stream of hints in session order, in the predictor API's format
  read_only_calls() recorded read-only tool calls (the shared Redis database is never changed)
  mcp_request()     one stateless MCP tools/call (pulse/agent/mcp_server.py runs stateless JSON: no session needed),
                    through the same URL / Host templates as the runners (MCP_URL_TEMPLATE, MCP_HOST_TEMPLATE)
  Recorder          every request -> requests.csv + summary.json in a run folder under results/load_runs/
                    (PULSE_LOAD_DIR overrides it)
"""

import csv
import json
import os
import statistics
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

from pulse import paths

COLD_S = 1.0                     # a tool call slower than this counts as cold (the runners' latency rule)
URL_TPL = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:8081/mcp")
HOST_TPL = os.getenv("MCP_HOST_TEMPLATE", "{svc}.default.example.com")


def parse_tasks(s) -> list[int]:
    return [int(t) for t in str(s or "").replace(",", " ").split()]


def episodes(seed: int = 252, tasks: list[int] | None = None, n: int = 46, fanout: bool = True) -> list[dict]:
    """The replay's episodes. Hints carry signals + history if data/tau-bench is there, else names only."""
    import pulse.experiments.replay as w1_replay
    a = SimpleNamespace(seed=seed, traces=None, tasks=tasks or None, predictor_url="hints", no_signals=False, repeat=1,
                        think_median=1.5, think_sigma=0.5, user_turn_median=8.0, user_turn_sigma=0.5, gap_scale=1.0,
                        rate=2.0, episodes=n, fanout=fanout, hint_timing="fair")
    try:
        return w1_replay.plan(a)
    except SystemExit as e:                    # no tau-bench conversations: hints without signals / history
        print(f"note: {e}; hints carry tool names only")
        a.predictor_url, a.episodes = None, n
        return w1_replay.plan(a)


def episode_hints(ep: dict, episode_id: str, workflow: str = "W1") -> list[dict]:
    """The hints of one episode, in order, as the agent sends them (see w1_replay.Replay.send_hint)."""
    out, traj = [], []
    for idx in ep["groups"]:
        for _, sig, hist in ep["hints"][idx[0]]:
            h = {"workflow_id": workflow, "episode_id": episode_id, "trajectory": traj[-8:],
                 "current_tool": traj[-1] if traj else "START", "ts": time.time()}
            if sig is not None:
                h["signals"] = sig
            if hist is not None:                # read by the LLM speculator and pulse_v3, ignored by the others
                h["history"] = hist
            out.append(h)
        traj += [ep["trace"]["calls"][j]["tool"] for j in idx]
    return out


def hint_stream(eps: list[dict], stream_id: str):
    """Endless hints: one episode after another (each with a fresh episode id), so a stateful predictor sees
    every session in order."""
    k, start = 0, zlib.crc32(stream_id.encode())          # crc32, not hash(): the same streams on every run
    while True:
        ep = eps[(start + k) % len(eps)]
        yield from episode_hints(ep, f"load-{stream_id}-{k}")
        k += 1


def read_only_calls(eps: list[dict]) -> list[tuple[str, dict]]:
    from pulse.agent.tools import READ_ONLY_TOOLS
    return [(c["tool"], c["args"]) for ep in eps for c in ep["trace"]["calls"] if c["tool"] in READ_ONLY_TOOLS]


def mcp_request(tool: str, args: dict) -> tuple[str, dict, dict]:
    """(url, headers, JSON body) of one MCP tools/call to the tool's Knative service."""
    svc = tool.replace("_", "-")
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if HOST_TPL:
        headers["Host"] = HOST_TPL.format(svc=svc, tool=tool)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}}
    return URL_TPL.format(svc=svc, tool=tool), headers, body


def mcp_failed(status: int, text: str) -> str:
    """'' if the tool answered, else why not (HTTP error, or a JSON-RPC error from the server)."""
    if status != 200:
        return f"HTTP {status}"
    return "JSON-RPC error" if '"error"' in text[:300] and '"result"' not in text[:300] else ""


def run_dir(target: str, engine: str, config: str, label: str = "") -> Path:
    base = Path(os.getenv("PULSE_LOAD_DIR") or paths.RESULTS / "load_runs")
    d = base / (f"{time.strftime('%Y-%m-%d_%H%M%S')}_load-{target}_{engine}_{config}" + (f"_{label}" if label else ""))
    d.mkdir(parents=True, exist_ok=True)
    print(f"-> {d}", flush=True)          # the UI's run log finds the result folder by this line
    return d


def _pct(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] * 1000, 1) if xs else None


def _stats(rows: list[tuple]) -> dict:
    lat = [r[4] for r in rows if r[5]]
    span = (max(r[0] + r[4] for r in rows) - min(r[0] for r in rows)) if rows else 0
    return {"requests": len(rows), "failures": sum(not r[5] for r in rows),
            "rps": round(len(rows) / span, 2) if span > 0 else None,
            "p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99),
            "mean": round(statistics.mean(lat) * 1000, 1) if lat else None, "max": _pct(lat, 100)}


def predictor_name(url: str | None) -> str | None:
    """The predictor answering at `url` (its GET /info), recorded with the run; None if there is none."""
    if not url:
        return None
    from pulse.predictor import client
    return client.info(url).get("name")


class Recorder:
    """Every request of a load run: (t_start, load, kind, name, latency_s, ok, error). `load` is what the run
    varies: users (Locust) or hints/s / burst size (asyncio). Tool calls slower than COLD_S count as cold."""

    def __init__(self, target: str, engine: str, config: str, params: dict):
        self.target, self.engine, self.config, self.params = target, engine, config, params
        self.rows: list[tuple] = []
        self.started = time.time()

    def add(self, t_start: float, load, kind: str, name: str, latency_s: float, ok: bool, error: str = ""):
        self.rows.append((t_start, load, kind, name, latency_s, ok, error))

    def write(self, out: Path, extra: dict | None = None) -> dict:
        with open(out / "requests.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_start", "load", "kind", "name", "latency_ms", "ok", "error"])
            for t, load, kind, name, lat, ok, err in self.rows:
                w.writerow([round(t, 3), load, kind, name, round(lat * 1000, 1), int(ok), err])
        tools = [r for r in self.rows if r[2] == "tool" and r[5]]
        stages = []
        for load in dict.fromkeys(r[1] for r in self.rows):             # in the order the run went through them
            rows = [r for r in self.rows if r[1] == load]
            stages.append({"load": load, **_stats(rows),
                           "cold_share": round(sum(r[4] > COLD_S for r in rows if r[2] == "tool") /
                                               max(1, sum(r[2] == "tool" and r[5] for r in rows)), 3)})
        s = _stats(self.rows)
        summary = {"kind": "load", "target": self.target, "engine": self.engine, "config": self.config,
                   "started": round(self.started, 3), "ended": round(time.time(), 3),
                   **{k: s[k] for k in ("requests", "failures", "rps")},
                   "latency_ms": {"all": {k: s[k] for k in ("p50", "p95", "p99", "mean", "max")}},
                   "cold_share": round(sum(r[4] > COLD_S for r in tools) / len(tools), 3) if tools else None,
                   "peak_load": max((r[1] for r in self.rows), default=None),
                   "by_kind": {k: _stats([r for r in self.rows if r[2] == k]) for k in sorted({r[2] for r in self.rows})},
                   "stages": stages, "params": self.params, **(extra or {})}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "run_config.json").write_text(json.dumps({"kind": "load", "target": self.target, "engine": self.engine,
                                                         "config": self.config, "params": self.params,
                                                         "mcp_url_template": URL_TPL, "mcp_host_template": HOST_TPL,
                                                         "runner": os.getenv("PULSE_RUNNER", "laptop")}, indent=2),
                                             encoding="utf-8")
        a = summary["latency_ms"]["all"]
        print(f"{self.target} ({self.engine}, {self.config}): {s['requests']} requests, {s['failures']} failed, "
              f"p50 {a['p50']} ms, p95 {a['p95']} ms, p99 {a['p99']} ms -> {out}", flush=True)
        return summary
