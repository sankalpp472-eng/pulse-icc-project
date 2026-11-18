"""
Regression tests for issue #35: a live run whose customer is a person is not written to a recording (the recording
holds the conversation text) unless the operator sets PULSE_RECORD_HUMAN=1. AI-customer runs are still recorded.
"""

import json

import pytest

from pulse.ui.session import Session


def _session(tmp_path, user):
    class Cluster:
        context = "kind-test"

        def episode_started(self, t0):
            pass

    s = Session({"kind": "live", "user": user}, cluster=Cluster(), sim=None, predictor_url=None, record_dir=tmp_path)
    s.state = "done"
    s.rec = {"id": "live-test", "user": user, "steps": [{"kind": "tool", "step": 1}], "first_text": "my card is 4111"}
    return s


@pytest.mark.parametrize("user,env,saved", [("ai", None, True), ("human", None, False), ("human", "1", True)])
def test_recording_gate(tmp_path, monkeypatch, user, env, saved):
    if env is None:
        monkeypatch.delenv("PULSE_RECORD_HUMAN", raising=False)
    else:
        monkeypatch.setenv("PULSE_RECORD_HUMAN", env)
    _session(tmp_path, user).save_recording()
    f = tmp_path / "live-test.json"
    assert f.exists() == saved
    if saved:
        assert json.loads(f.read_text(encoding="utf-8"))["id"] == "live-test"
