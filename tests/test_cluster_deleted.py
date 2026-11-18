"""
Regression tests for issue #28: after the kind cluster is deleted (Setup's Delete cluster, or `kind delete cluster`
in a terminal) the UI must stop reporting it: the top bar falls back to the simulated cluster and says why, and the
context it shows is the current one. The Setup tab's own part (the button confirms in the page, every outcome is
written to the Log panel, the panel reloads when the cluster changes) is in pulse/ui/static/app.js.

No cluster needed: kubectl's answers are faked.
"""

import pytest

import pulse.ui.backend as backend


def kubectl_context(monkeypatch, answer):
    """Fake `kubectl config current-context`: answer is the context, or None for 'current-context is not set'."""
    def run(args, timeout=10):
        if args[:3] == ["kubectl", "config", "current-context"]:
            return (0, answer + "\n") if answer else (1, "error: current-context is not set\n")
        return 1, "not faked"
    monkeypatch.setattr(backend, "_run", run)


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.setattr(backend.Cluster, "_poll", lambda self: None)     # no background thread in tests
    c = backend.Cluster()
    c.mode, c.context, c.reachable, c.tools_found = "knative", "kind-pulse-kn", True, 14
    c.pods = [{"tool": "get_user_details", "ready": 1, "starting": 0, "busy": 0}]
    return c


def test_deleted_cluster_is_noticed(cluster, monkeypatch):
    kubectl_context(monkeypatch, None)                  # kind delete cluster removed the context
    assert cluster.check_context() is True
    assert cluster.context is None and not cluster.reachable and cluster.pods == []
    assert "kind-pulse-kn" in cluster.lost and "simulated" in cluster.lost


def test_context_switched_elsewhere_is_noticed(cluster, monkeypatch):
    kubectl_context(monkeypatch, "arn:aws:eks:ap-south-1:1:cluster/pulse-eks")
    assert cluster.check_context() is True
    assert cluster.context.endswith("pulse-eks") and "now points at" in cluster.lost


def test_unchanged_context_is_left_alone(cluster, monkeypatch):
    kubectl_context(monkeypatch, "kind-pulse-kn")
    assert cluster.check_context() is False
    assert cluster.reachable and cluster.lost is None


@pytest.mark.parametrize("failure", [(1, "FileNotFoundError"), (1, "TimeoutExpired"),
                                     (1, "error: open /home/u/.kube/config: permission denied")])
def test_no_answer_from_kubectl_changes_nothing(cluster, monkeypatch, failure):
    """kubectl missing (e.g. in CI), hung or failing is not 'the cluster was deleted'."""
    monkeypatch.setattr(backend, "_run", lambda args, timeout=10: failure)
    assert cluster.check_context() is False
    assert cluster.context == "kind-pulse-kn" and cluster.reachable and cluster.lost is None


def test_simulated_mode_just_tracks_the_context(cluster, monkeypatch):
    cluster.mode = "sim"
    kubectl_context(monkeypatch, None)
    assert cluster.check_context() is False
    assert cluster.context is None and cluster.lost is None   # the top bar's "Knative (...)" label empties


def test_ui_falls_back_to_the_simulator(monkeypatch):
    pytest.importorskip("fastapi")
    import pulse.ui.server as server
    monkeypatch.setattr(backend.Services, "status", lambda self: {})
    monkeypatch.setattr(backend.Services, "stop", lambda self, name=None: {})
    monkeypatch.setattr(backend.Services, "match_cluster", lambda self: True)
    server.S["session"] = server.S["run"] = None
    c = server.cluster
    c.mode, c.context, c.reachable = "knative", "kind-pulse-kn", True
    kubectl_context(monkeypatch, None)
    try:
        c.check_context()
        s = server.state()
        assert s["cluster"]["mode"] == "sim" and s["cluster"]["context"] is None
        assert "simulated" in s["cluster_lost"]
        assert server.state()["cluster_lost"] is None                       # said once
    finally:
        c.mode, c.lost = "sim", None
