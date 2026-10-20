"""
Checks for step 9 (predictor -> PrewarmRequest objects). No cluster needed (dry-run mode).

    python -m pytest predictor/tests -q          # from the repo root
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from crd_writer import CRDWriter, make_object, service_name  # noqa: E402


def test_service_name_matches_knative():
    assert service_name("search_direct_flight") == "search-direct-flight"


def test_object_has_the_agreed_fields():
    o = make_object("search_direct_flight", 0.6234567, 0.3, 60, "task1-1", "W1")
    assert o["apiVersion"] == "pulse.iisc.ac.in/v1alpha1" and o["kind"] == "PrewarmRequest"
    assert o["metadata"]["name"] == "search-direct-flight"
    s = o["spec"]
    assert s["tool"] == "search_direct_flight" and s["service"] == "search-direct-flight"
    assert s["probability"] == 0.6235 and s["threshold"] == 0.3
    assert s["ttlSeconds"] == 60 and s["desiredReplicas"] == 1
    assert s["requestedAt"].endswith("Z") and s["episodeId"] == "task1-1" and s["workflowId"] == "W1"


def test_dryrun_writes_one_line_per_tool(tmp_path, monkeypatch):
    out = tmp_path / "pwr.jsonl"
    monkeypatch.setenv("CRD_DRYRUN_FILE", str(out))
    w = CRDWriter("dryrun")
    w.submit([("get_reservation_details", 0.76), ("search_direct_flight", 0.31)], 0.3, 60, "ep", "W1")
    assert w.flush()
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["spec"]["tool"] for x in lines] == ["get_reservation_details", "search_direct_flight"]
    assert w.stats["written"] == 2 and w.stats["failed"] == 0


def test_off_mode_does_nothing():
    w = CRDWriter("off")
    w.submit([("calculate", 0.9)], 0.3, 60)
    assert w.stats["written"] == 0
