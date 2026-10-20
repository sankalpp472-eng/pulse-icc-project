"""
Prometheus metrics published by the experiment runners (w1_live.py, later the replay).

The runner serves them at http://<laptop>:<port>/metrics; Prometheus in the kind cluster
scrapes host.docker.internal:9105 (see deploy/monitoring/values.yaml).

Every series carries run_id, so several runs can be told apart in Prometheus/Grafana.
All tool x cold/warm series are created at start with value 0, so increase() over a
run counts the first call too.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 60)

TOOL_CALL_SECONDS = Histogram(
    "pulse_tool_call_seconds", "Latency of one MCP tool call, as seen by the agent",
    ["run_id", "workload", "tool", "temperature"], buckets=LATENCY_BUCKETS)
TOOL_CALLS = Counter(
    "pulse_tool_calls_total", "MCP tool calls", ["run_id", "workload", "tool", "temperature"])
TOOL_IDLE_GAP_SECONDS = Histogram(
    "pulse_tool_idle_gap_seconds", "Time since the same tool's previous call ended",
    ["run_id", "workload", "temperature"], buckets=(1, 2, 5, 10, 15, 30, 45, 60, 75, 90, 120, 180, 300, 600))
LLM_THINK_SECONDS = Histogram(
    "pulse_llm_think_seconds", "LLM response time before a tool call (the prewarm window)",
    ["run_id", "workload"], buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 20))
RATE_LIMIT_WAIT_SECONDS = Counter(
    "pulse_rate_limit_wait_seconds_total", "Seconds spent waiting on LLM rate limits", ["run_id", "workload"])
EPISODES = Counter(
    "pulse_episodes_total", "Finished agent episodes", ["run_id", "workload", "status"])
CURRENT_TASK = Gauge(
    "pulse_current_task", "tau-bench task id being run (-1 = idle)", ["run_id", "workload"])
RUN_INFO = Gauge(
    "pulse_run_info", "Constant 1; labels describe the run",
    ["run_id", "workload", "config", "window", "agent_model", "user_model", "database"])


class RunMetrics:
    def __init__(self, port: int, run_id: str, workload: str, tools: list[str], info: dict):
        self.on = port > 0
        self.l = {"run_id": run_id, "workload": workload}
        if not self.on:
            return
        start_http_server(port)
        RUN_INFO.labels(**self.l, **info).set(1)
        CURRENT_TASK.labels(**self.l).set(-1)
        for tool in tools:
            for temp in ("cold", "warm"):
                TOOL_CALLS.labels(**self.l, tool=tool, temperature=temp)
                TOOL_CALL_SECONDS.labels(**self.l, tool=tool, temperature=temp)
        for temp in ("cold", "warm"):
            TOOL_IDLE_GAP_SECONDS.labels(**self.l, temperature=temp)
        LLM_THINK_SECONDS.labels(**self.l)
        RATE_LIMIT_WAIT_SECONDS.labels(**self.l)

    def task(self, task_id: int):
        if self.on:
            CURRENT_TASK.labels(**self.l).set(task_id)

    def call(self, tool: str, cold: bool, latency_s: float, idle_gap_s: float | None,
             think_s: float | None, rate_wait_s: float | None):
        if not self.on:
            return
        temp = "cold" if cold else "warm"
        TOOL_CALLS.labels(**self.l, tool=tool, temperature=temp).inc()
        TOOL_CALL_SECONDS.labels(**self.l, tool=tool, temperature=temp).observe(latency_s)
        if idle_gap_s is not None:
            TOOL_IDLE_GAP_SECONDS.labels(**self.l, temperature=temp).observe(idle_gap_s)
        if think_s:
            LLM_THINK_SECONDS.labels(**self.l).observe(think_s)
        if rate_wait_s:
            RATE_LIMIT_WAIT_SECONDS.labels(**self.l).inc(rate_wait_s)

    def episode(self, status: str):
        if self.on:
            EPISODES.labels(**self.l, status=status.split(":")[0]).inc()
            CURRENT_TASK.labels(**self.l).set(-1)
