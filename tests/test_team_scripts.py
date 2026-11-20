"""
The team experiment scripts' Python parts (experiments/team/): the in-cluster cold-start runner, the Prometheus
export's run windows, the summary of every kind of run, and the model fingerprint. No cluster needed: the tool
backend and the pod watcher are stubs.
"""

import csv
import json

import pytest

from pulse import paths


class FakeBackend:
    names = ["get_user_details", "think"]

    def __init__(self):
        self.calls = 0

    def call(self, name, args):
        self.calls += 1
        return ("ok", 1.5 if self.calls % 2 else 0.04)       # cold, then warm


class FakeWatcher:
    def __init__(self, *a, **k):
        self.stop_flag = False

    def start(self):
        pass

    def new_pod_during(self, tool, t0, t1):
        return f"{tool}-pod"

    def pod_rows(self):
        return [{"pod": f"{t}-pod", "tool": t, "created": 0, "scheduled": 0.0, "containers_started": 0.8,
                 "ready": 1.4, "first_seen": 0, "last_seen": 5} for t in FakeBackend.names]


def test_coldstart_records_cold_warm_and_startup(tmp_path, monkeypatch):
    import pulse.experiments.coldstart as cs
    monkeypatch.setattr(cs, "make_tool_backend", FakeBackend)
    monkeypatch.setattr(cs, "PodWatcher", FakeWatcher)
    monkeypatch.setattr(cs, "wait_all_zero", lambda w: None)
    monkeypatch.setattr(cs, "run_dir", lambda *a: tmp_path)
    monkeypatch.setattr(cs.common, "run_config", lambda *a, **k: {})
    monkeypatch.setattr(cs.common, "reset_db", lambda: None)
    # only this module's clock: patching time.sleep itself would turn other tests' background threads into busy loops
    import time as real_time
    monkeypatch.setattr(cs, "time", type("T", (), {"time": staticmethod(real_time.time), "sleep": staticmethod(lambda s: None)}))
    monkeypatch.setattr("sys.argv", ["coldstart", "--reps", "2"])
    assert cs.main() == 0
    rows = list(csv.DictReader(open(tmp_path / "coldstart.csv", encoding="utf-8")))
    assert len(rows) == 4 and {r["tool"] for r in rows} == set(FakeBackend.names)
    assert all(float(r["cold_ms"]) == 1500 and float(r["warm_ms"]) == 40 for r in rows)
    assert all(float(r["readiness_s"]) == pytest.approx(0.6) for r in rows)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["cold_ms"]["median"] == 1500 and s["per_tool"]["think"]["rounds"] == 2

    # the summary reads it as a cold-start run
    import pulse.experiments.team_summary as ts
    man = tmp_path / "manifest.tsv"
    man.write_text(f"coldstart\tairline\treps2\t1\tdone\t{tmp_path}\t0\t0\n")
    monkeypatch.setattr("sys.argv", ["ts", "--manifest", str(man), "--exp", "coldstart", "--out", str(tmp_path / "out")])
    assert ts.main() == 0
    md = (tmp_path / "out" / "coldstart.md").read_text(encoding="utf-8")
    assert "Median cold 1500" in md and "| think | 1500" in md


def test_prom_export_takes_the_finished_runs_windows(tmp_path):
    from pulse.experiments.prom_export import windows
    man = tmp_path / "m.tsv"
    man.write_text("ttl\tairline\tttl30\t1\tdone\tresults/x\t1000\t2000\n"
                   "ttl\tairline\tttl60\t1\tfailed(rc=1)\tpod\t3000\t4000\n"
                   "topk\tairline\ttop1\t1\tdone\tresults/y\t5000\t6000\n")
    assert windows(man, "ttl", 60) == [("airline/ttl30/t1", 940.0, 2060.0)]


def test_the_summary_adds_resource_use_from_prometheus(tmp_path):
    from pulse.experiments.team_summary import resource_metrics
    prom = tmp_path / "prometheus"
    prom.mkdir()
    def write(name, rows):
        with open(prom / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "time", "labels", "value"])
            w.writerows(rows)
    run = "airline/C0/t1"
    tool = json.dumps({"pod": "think-00001-deployment-abc", "container": "user-container"})
    pred = json.dumps({"pod": "pulse-predictor-xyz", "container": "predictor"})
    write("pod_cpu_cores", [(run, 0, tool, 0.1), (run, 15, tool, 0.1), (run, 0, pred, 0.2), ("other/x/t1", 0, tool, 9)])
    write("pod_memory_bytes", [(run, 0, tool, 2**20 * 50), (run, 15, tool, 2**20 * 70)])
    write("tool_pods", [(run, 0, "{}", 2), (run, 15, "{}", 3)])
    write("node_cpu_busy", [(run, 0, "{}", 0.2), (run, 15, "{}", 0.4)])
    m = resource_metrics(prom, run)
    assert m == {"tool_cpu_core_s": 3.0, "tool_mem_mib": 60.0, "pulse_cpu_core_s": 3.0, "node_cpu_busy_pct": 30.0,
                 "tool_pods_peak": 3.0}


def test_model_fingerprint_is_stable():
    if not (paths.MODELS / "action_model.pkl").exists():
        pytest.skip("no airline model")
    from pulse.experiments.model_fingerprint import fingerprint
    assert fingerprint("airline") == fingerprint("airline")


def test_controller_events_printed_onto_one_line_are_all_read(tmp_path):
    from pulse.experiments.team_summary import json_events
    f = tmp_path / "controller.jsonl"
    f.write_text('{"event": "warmed", "tool": "a"}{"event": "keepalive", "tool": "b"}\n'
                 'not json\n{"event": "skipped"} {"event": "evicted"}\n', encoding="utf-8")
    assert [e["event"] for e in json_events(f)] == ["warmed", "keepalive", "skipped", "evicted"]
