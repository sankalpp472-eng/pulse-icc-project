"""
One interactive demo episode, run in a thread and shown step by step in the UI.

  live    the ReAct agent (agent/agent.py helpers) talks to a customer:
            ai     an LLM plays the customer from a tau-bench task (agent/user.py SimulatedUser)
            human  you type the customer's messages in the browser
  replay  a recorded tau-bench conversation's tool calls, with seeded think / customer-reply gaps and
          fair hints (one per LLM step), like test/test_knative/w1_replay.py, but one episode, on screen

Before every LLM step it sends the predictor a hint (trajectory + signals, as agent/hints.py does) and shows
the answer. The tools run either on the real cluster (MCP, Knative: the controller prewarms) or on the
simulated cluster (ui/sim_cluster.py: the session itself applies the predictor's prewarm list).
"""

import asyncio
import json
import math
import queue
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("agent", "predictor", "test/test_knative"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

COLD_LATENCY_S = 1.0        # on the real cluster a call slower than this counts as cold (as the runners do)
HISTORY_N = 10              # messages the LLM speculator reads


class Stopped(Exception):
    pass


def compact(messages: list[dict]) -> list[dict]:
    """The conversation as the LLM speculator reads it (baselines/llm_speculator/steps.py format)."""
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant", "tool"):
            continue
        cm = {"role": role, "content": str(m.get("content") or "")[:600]}
        if role == "tool":
            cm["name"] = m.get("name", "")
        if role == "assistant" and m.get("tool_calls"):
            cm["calls"] = [[tc["function"]["name"], tc["function"].get("arguments")] for tc in m["tool_calls"]]
        out.append(cm)
    return out[-HISTORY_N:]


# ====================================================================== tool backends
class SimTools:
    """The tools on the simulated cluster: cold-start delays from SimCluster, real results from the local DB."""

    def __init__(self, sim):
        from tools import LocalToolBackend
        self.local, self.sim = LocalToolBackend(), sim
        self.schemas = self.local.schemas

    def run(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        wait, _, pod = self.sim.acquire(name)
        try:
            time.sleep(wait + 0.04)
            out, _ = self.local.call(name, args)
        finally:
            self.sim.release(pod)
        lat = time.perf_counter() - t0
        # cold = the same rule as the runners (slower than 1 s); a shorter wait = a prewarm that almost made it
        return {"out": out, "latency_s": lat, "cold": lat > COLD_LATENCY_S, "wait_s": wait}


class KnativeTools:
    """The tools on the real cluster, over MCP (Knative's activator holds a call during a cold start)."""

    def __init__(self, env: dict):
        import os
        os.environ.update(env)
        from tools import MCPToolBackend
        self.mcp = MCPToolBackend()
        self.schemas = self.mcp.schemas

    def run(self, name: str, args: dict) -> dict:
        out, lat, _ = asyncio.run(self.mcp.acall(name, args))
        return {"out": out, "latency_s": lat, "cold": lat > COLD_LATENCY_S, "wait_s": None}


# ====================================================================== the session
class Session(threading.Thread):
    def __init__(self, cfg: dict, cluster, sim, predictor_url: str | None, record_dir: Path | None = None):
        super().__init__(daemon=True)
        self.cfg, self.cluster, self.sim, self.predictor_url = cfg, cluster, sim, predictor_url
        self.record_dir = record_dir
        self.rec: dict | None = None              # live episodes: the trajectory with its timing, saved at the end
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.state = "starting"
        self.stop_flag = False
        self.inbox: queue.Queue = queue.Queue()     # the human customer's messages
        self.t0 = time.time()
        self.prewarmed: dict[int, dict] = {}         # LLM step -> {tool: k} the predictor asked for
        self.calls: list[dict] = []
        self.hints = {"sent": 0, "answered": 0, "rtt_ms": []}
        self.episode_id = f"ui-{int(self.t0 * 1000)}"
        # end-to-end clock: from the first message of the conversation to the end of the episode (setup excluded)
        self.t_start = self.t_end = None
        self.tool_wait = 0.0                         # time the agent was blocked on tools (a parallel step = its slowest call)
        self.rng = random.Random(int(cfg.get("seed", 252)))
        self.pod_s0 = sim.live_pod_seconds() if sim else None

    # ------------------------------------------------------------------ plumbing
    def emit(self, type_: str, **kw):
        with self.lock:
            self.events.append({"seq": len(self.events), "t": round(time.time() - self.t0, 2), "type": type_, **kw})

    def view(self, after: int = 0) -> dict:
        with self.lock:
            return {"id": self.episode_id, "state": self.state, "cfg": self.cfg, "events": self.events[after:],
                    "n": len(self.events), "t_start": self.t_start, "t_end": self.t_end,
                    "tool_wait_s": round(self.tool_wait, 3)}

    def begin(self):
        """Start the end-to-end clock (the conversation starts now)."""
        self.t_start = time.time()
        self.emit("clock", state="start")

    def stop(self):
        self.stop_flag = True
        self.inbox.put(None)

    def sleep(self, s: float):
        end = time.time() + max(0.0, s)
        while time.time() < end:
            if self.stop_flag:
                raise Stopped()
            time.sleep(min(0.1, end - time.time()))

    def check(self):
        if self.stop_flag:
            raise Stopped()

    # ------------------------------------------------------------------ hints
    def hint(self, step: int, kind: str, trajectory: list[str], signals: dict | None, history: list | None):
        """Send one hint (in the background, the agent never waits) and show the predictor's answer."""
        if not self.predictor_url:
            return
        hint = {"workflow_id": "W1", "episode_id": self.episode_id, "trajectory": list(trajectory)[-8:],
                "current_tool": trajectory[-1] if trajectory else "START", "ts": time.time()}
        if signals is not None:
            hint["signals"] = signals
        if history is not None and self.cfg.get("predictor") == "speculator":
            hint["history"] = history
        self.hints["sent"] += 1
        self.emit("hint", step=step, kind=kind)

        def post():
            t = time.perf_counter()
            try:
                req = urllib.request.Request(self.predictor_url + "/predict", data=json.dumps(hint).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=15) as r:
                    ans = json.loads(r.read())
            except Exception as e:
                self.emit("prediction", step=step, kind=kind, error=f"{type(e).__name__}: {e}"[:160])
                return
            rtt = round((time.perf_counter() - t) * 1000, 1)
            self.hints["answered"] += 1
            self.hints["rtt_ms"].append(rtt)
            warm = {p["tool"]: int(p.get("k") or 1) for p in ans.get("prewarm", [])}
            self.prewarmed.setdefault(step, {}).update(warm)
            self.emit("prediction", step=step, kind=kind, rtt_ms=rtt, model=ans.get("model"),
                      theta=ans.get("theta"), top=[{"tool": x["tool"], "p": round(x["p"], 3)} for x in ans["ranked"][:4]],
                      prewarm=[{"tool": t, "k": k} for t, k in warm.items()],
                      signals={k: v for k, v in (signals or {}).items() if v} if signals else None)
            if self.sim:                       # the simulated controller: start the pods right away
                for t, k in warm.items():
                    p = next((x["p"] for x in ans["prewarm"] if x["tool"] == t), None)
                    self.sim.prewarm(t, k, p)

        threading.Thread(target=post, daemon=True).start()

    # ------------------------------------------------------------------ tool calls
    def run_calls(self, step: int, calls: list[tuple[str, dict]], tools, parallel: bool) -> list[str]:
        """Run one step's calls (at once if parallel), show each with latency / cold / prewarmed."""
        results: list[dict | None] = [None] * len(calls)

        def one(i, name, args):
            r = tools.run(name, args)
            r["t_start"] = time.time() - r["latency_s"]
            results[i] = r

        if parallel and len(calls) > 1:
            ts = [threading.Thread(target=one, args=(i, n, a)) for i, (n, a) in enumerate(calls)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        else:
            for i, (n, a) in enumerate(calls):
                self.check()
                one(i, n, a)
        outs = []
        lats = [r["latency_s"] for r in results if r]
        if lats:                                   # the agent waits for the whole step: parallel = slowest, else the sum
            self.tool_wait += max(lats) if parallel and len(calls) > 1 else sum(lats)
        warm = self.prewarmed.get(step, {})
        for (name, args), r in zip(calls, results):
            rec = {"step": step, "tool": name, "args": args, "latency_ms": round(r["latency_s"] * 1000, 1),
                   "cold": r["cold"], "wait_ms": None if r["wait_s"] is None else round(r["wait_s"] * 1000),
                   "prewarmed": name in warm, "parallel": len(calls) if parallel and len(calls) > 1 else 1,
                   "result": r["out"][:300]}
            self.calls.append(rec)
            self.emit("call", **rec)
            outs.append(r["out"])
        return outs

    # ------------------------------------------------------------------ main
    def run(self):
        try:
            self.state = "running"
            if self.cfg["kind"] == "replay":
                self.run_replay()
            elif self.cfg["kind"] == "recorded":
                self.run_recorded()
            else:
                self.run_live()
            self.state = "done"
        except Stopped:
            self.state = "stopped"
            self.emit("info", text="Stopped.")
        except Exception as e:
            self.state = "failed"
            self.emit("error", text=f"{type(e).__name__}: {e}"[:600])
        finally:
            if self.t_start and not self.t_end:
                self.t_end = time.time()
            self.save_recording()
            self.emit("end", summary=self.summary())

    def save_recording(self):
        """A live episode with at least one tool call -> record_dir/<id>.json, replayable with kind=recorded."""
        r = self.rec
        if not (r and self.record_dir and any(s["kind"] == "tool" for s in r["steps"])):
            return
        r.update(state=self.state, live_e2e_s=round(self.t_end - self.t_start, 2) if self.t_start and self.t_end else None,
                 live_tool_wait_s=round(self.tool_wait, 2),
                 live_cold=sum(c["cold"] for c in self.calls), live_calls=len(self.calls),
                 config="C3" if self.cfg.get("predictor") else "C0", policy=self.cfg.get("policy_label"),
                 cluster="simulated" if self.sim else self.cluster.context)
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / f"{r['id']}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            self.recording_id = r["id"]
            self.emit("recorded", recording=r["id"])
        except OSError as e:
            self.emit("error", text=f"could not save the recording: {e}")

    def tools(self):
        if self.sim:
            return SimTools(self.sim)
        return KnativeTools(self.cluster.tool_env())

    def summary(self) -> dict:
        n = len(self.calls)
        lat = sorted(c["latency_ms"] for c in self.calls)
        cold = [c for c in self.calls if c["cold"]]
        rtt = sorted(self.hints["rtt_ms"])
        return {"calls": n, "cold": len(cold), "cold_share": round(len(cold) / n, 3) if n else None,
                "prewarmed_hits": sum(c["prewarmed"] for c in self.calls),
                "cold_wait_s": round(sum(c["latency_ms"] for c in cold) / 1000, 2),
                "mean_ms": round(sum(lat) / n, 1) if n else None,
                "p95_ms": lat[min(n - 1, int(0.95 * n))] if n else None,
                "hints_sent": self.hints["sent"], "hints_answered": self.hints["answered"],
                "hint_rtt_p50_ms": rtt[len(rtt) // 2] if rtt else None,
                "pod_seconds": round(self.sim.live_pod_seconds() - self.pod_s0, 1) if self.sim else None,
                "duration_s": round(time.time() - self.t0, 1),
                "e2e_s": round(self.t_end - self.t_start, 2) if self.t_start and self.t_end else None,
                "tool_wait_s": round(self.tool_wait, 2), "state": self.state,
                "seed": self.cfg.get("seed", 252), "trial": self.cfg.get("trial"), "speed": self.cfg.get("speed"),
                "kind": self.cfg["kind"], "user": self.cfg.get("user") if self.cfg["kind"] == "live" else None, "predictor": self.cfg.get("predictor") or "none",
                "policy": self.cfg.get("policy_label") if self.cfg.get("predictor") else None,
                "cluster": "simulated" if self.sim else self.cluster.context, "task": self.cfg.get("task_id"),
                "recording": self.cfg.get("recording") or getattr(self, "recording_id", None)}

    # ------------------------------------------------------------------ live: agent + customer
    def run_live(self):
        import llm
        from agent import _assistant_msg, _chat_within_limit
        from prompts import AGENT_SYSTEM_PROMPT
        from tools import READ_ONLY_TOOLS, TERMINATE_TOOLS
        from user import STOP, SimulatedUser
        from signals import compute_signals, events_from_messages
        from tau_airline import load_tasks

        llm.DEFAULT_MODEL = self.cfg.get("agent_model") or llm.DEFAULT_MODEL
        tasks = {t["id"]: t for t in load_tasks()}
        task = tasks.get(int(self.cfg["task_id"])) if self.cfg.get("task_id") not in (None, "") else None
        goal = task["instruction"] if task else (self.cfg.get("goal") or "")
        tools = self.tools()
        self.emit("info", text=f"Agent {llm.DEFAULT_MODEL} · customer: "
                               f"{'AI (' + self.cfg.get('user_model', '') + ')' if self.cfg['user'] == 'ai' else 'you'}"
                               f" · tools: {'simulated cluster' if self.sim else self.cluster.context}")
        if goal:
            self.emit("goal", text=goal, task_id=task["id"] if task else None,
                      gold=[a["name"] for a in task["actions"]] if task else None)

        if self.cfg["user"] == "ai":
            if not goal:
                raise ValueError("the AI customer needs a task (pick one)")
            customer = SimulatedUser(goal)
            customer.model = self.cfg.get("user_model") or "openai/gpt-oss-20b"
            median = float(self.cfg.get("user_delay", 0) or 0)

            def say(agent_text: str | None) -> str:
                t = time.time()
                self.emit("typing", who="customer")
                text = customer.start() if agent_text is None else customer.reply(agent_text)
                d = 0.0 if median <= 0 or text.strip() == STOP else \
                    min(60.0, max(1.0, self.rng.lognormvariate(math.log(median), 0.5)))
                self.sleep(d - (time.time() - t))
                return text
        else:
            def say(agent_text: str | None) -> str:
                self.emit("waiting_for_user")
                self.state = "waiting"
                text = self.inbox.get()
                self.state = "running"
                if text is None:
                    raise Stopped()
                return text

        # the recording: everything a replay needs to reproduce this conversation's timeline exactly
        self.rec = {"id": f"live-{time.strftime('%Y%m%d-%H%M%S')}", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "task_id": task["id"] if task else None, "goal": goal, "agent_model": llm.DEFAULT_MODEL,
                    "user": self.cfg["user"], "user_model": self.cfg.get("user_model") if self.cfg["user"] == "ai" else None,
                    "user_delay": self.cfg.get("user_delay"), "steps": []}
        self.begin()
        first = say(None)
        self.rec.update(first_user_s=round(time.time() - self.t_start, 3), first_text=first)
        self.emit("customer", text=first)
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": first}]
        trajectory: list[str] = []
        for step in range(1, int(self.cfg.get("max_steps", 30)) + 1):
            self.check()
            try:
                sig = compute_signals(events_from_messages(messages, trajectory))
            except Exception:
                sig = None
            hist = compact(messages)
            self.hint(step, "step", trajectory, sig, hist)
            self.emit("thinking", step=step)
            r, _ = _chat_within_limit(messages, tools.schemas)
            m = r.message
            st = {"step": step, "think_s": round(r.latency_s, 3), "signals": sig, "history": hist}
            self.rec["steps"].append(st)
            messages.append(_assistant_msg(m))
            self.check()
            if m.tool_calls:
                parsed = []
                for tc in m.tool_calls:
                    try:
                        parsed.append((tc, json.loads(tc.function.arguments or "{}")))
                    except json.JSONDecodeError:
                        parsed.append((tc, None))
                ok = [(tc.function.name, a) for tc, a in parsed if a is not None]
                par = len(ok) > 1 and all(n in READ_ONLY_TOOLS for n, _ in ok)
                st.update(kind="tool", calls=[{"tool": n, "args": a} for n, a in ok], parallel=par,
                          text=(m.content or "")[:300])
                self.emit("agent_calls", step=step, think_s=round(r.latency_s, 2), tools=[n for n, _ in ok],
                          text=(m.content or "")[:300])
                outs = iter(self.run_calls(step, ok, tools, par))
                for tc, a in parsed:
                    obs = next(outs) if a is not None else "Error: arguments were not valid JSON"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
                    trajectory.append(tc.function.name)
                if any(n in TERMINATE_TOOLS for n, _ in ok):
                    self.emit("info", text="Transferred to a human agent: episode over.")
                    return
            else:
                text = m.content or ""
                self.emit("agent", step=step, text=text, think_s=round(r.latency_s, 2))
                st.update(kind="respond", text=text)
                t_say = time.time()
                reply = say(text)
                st.update(reply=reply, user_s=round(time.time() - t_say, 3))
                if reply.strip() != STOP:
                    reply = reply.replace(STOP, "").strip()
                if reply.strip() == STOP or not reply:
                    self.emit("info", text="The customer ended the conversation.")
                    return
                self.emit("customer", text=reply)
                messages.append({"role": "user", "content": reply})
        self.emit("info", text="Reached the step limit.")

    # ------------------------------------------------------------------ replay: a recorded episode
    def run_replay(self):
        import w1_replay
        task_id = int(self.cfg["task_id"])
        traces = [t for t in w1_replay.load_traces(None) if t["task_id"] == task_id and t["calls"]]
        if not traces:
            raise ValueError(f"no recorded tau-bench run with tool calls for task {task_id}")
        run_no = int(self.cfg.get("trial", 0)) % len(traces)          # the UI's "recorded run #" (0-based)
        tr = traces[run_no]
        if self.predictor_url:
            w1_replay.attach_signals([tr])
        groups = w1_replay.fanout_groups(tr) if self.cfg.get("fanout", True) else [[i] for i in range(len(tr["calls"]))]
        think_m, user_m = float(self.cfg.get("think_median", 1.5)), float(self.cfg.get("user_turn_median", 8.0))
        speed = float(self.cfg.get("speed", 1.0))
        draw = lambda med, lo, hi: min(hi, max(lo, self.rng.lognormvariate(math.log(med), 0.5)))
        tools = self.tools()
        self.emit("info", text=f"Replaying recorded run #{run_no + 1} of task {task_id} ({tr['source_model']}): "
                               f"{len(tr['calls'])} tool calls in {len(groups)} tool steps · think ~{think_m}s · "
                               f"customer ~{user_m}s{f' · gaps x{speed}' if speed != 1 else ''}")
        trajectory: list[str] = []
        step = 0
        self.begin()
        for g in groups:
            c = tr["calls"][g[0]]
            n = c["user_turns_before"]
            states = c.get("hint_states") or [(c.get("signals"), c.get("history"))] * (n + 1)
            for k in range(n):                                   # the agent answers, the customer replies
                step += 1
                sig, hist = states[k] if k < len(states) else states[0]
                self.hint(step, "reply", trajectory, sig, hist)
                self.emit("agent", step=step, text="(replies to the customer; recorded conversation)", think_s=None)
                gap = draw(user_m, 1.0, 60.0) * speed
                self.emit("customer", text=f"(customer replies after {gap:.1f} s)")
                self.sleep(gap)
            step += 1
            sig, hist = states[-1]
            self.hint(step, "call", trajectory, sig, hist)
            think = draw(think_m, 0.2, 15.0) * speed
            self.emit("thinking", step=step, think_s=round(think, 2))
            self.sleep(think)
            calls = [(tr["calls"][i]["tool"], tr["calls"][i]["args"]) for i in g]
            self.emit("agent_calls", step=step, think_s=round(think, 2), tools=[n for n, _ in calls], text="")
            self.run_calls(step, calls, tools, parallel=len(g) > 1)
            trajectory += [n for n, _ in calls]
        self.emit("info", text="Recorded episode finished.")

    # ------------------------------------------------------------------ replay of one of OUR live runs
    def run_recorded(self):
        """The exact trajectory of a live episode recorded by this UI: the same customer messages and reply times,
        the same agent think times and messages, the same tool calls (parallel where the agent ran them at once)
        and the same hints. Only the tools' latencies differ, so C0 vs C3 compares prewarming and nothing else."""
        path = self.record_dir / f"{self.cfg['recording']}.json" if self.record_dir else None
        if not path or not path.exists():
            raise ValueError(f"recording {self.cfg.get('recording')} not found")
        rec = json.loads(path.read_text(encoding="utf-8"))
        tools = self.tools()
        self.emit("info", text=f"Replaying live run {rec['id']} (task {rec.get('task_id')}, agent {rec.get('agent_model')}, "
                               f"customer {rec.get('user')}; recorded as {rec.get('config')}: "
                               f"{rec.get('live_e2e_s')} s end-to-end): same messages, timing and tool calls")
        if rec.get("goal"):
            self.emit("goal", text=rec["goal"], task_id=rec.get("task_id"))
        trajectory: list[str] = []
        self.begin()
        self.emit("typing", who="customer")
        self.sleep(rec.get("first_user_s", 0))
        self.emit("customer", text=rec.get("first_text", ""))
        for st in rec["steps"]:
            self.check()
            step = st["step"]
            self.hint(step, "step", trajectory, st.get("signals"), st.get("history"))
            self.emit("thinking", step=step, think_s=st.get("think_s"))
            self.sleep(st.get("think_s", 0))
            if st.get("kind") == "tool":
                calls = [(c["tool"], c["args"]) for c in st["calls"]]
                self.emit("agent_calls", step=step, think_s=st.get("think_s"), tools=[n for n, _ in calls],
                          text=st.get("text", ""))
                self.run_calls(step, calls, tools, parallel=bool(st.get("parallel")))
                trajectory += [n for n, _ in calls]
            elif st.get("kind") == "respond":
                self.emit("agent", step=step, text=st.get("text", ""), think_s=st.get("think_s"))
                if st.get("reply") is None:
                    break
                self.emit("typing", who="customer")
                self.sleep(st.get("user_s", 0))
                reply = st["reply"].strip()
                if reply == "###STOP###" or not reply.replace("###STOP###", "").strip():
                    self.emit("info", text="The customer ended the conversation.")
                    break
                self.emit("customer", text=reply.replace("###STOP###", "").strip())
        self.emit("info", text="End of the recorded live run.")
