"""
A fingerprint of the trained lightgbm / markov models: what they PREDICT, not their file bytes (a pickle of the
same model differs from run to run). sha256 over the gbt and ngram probabilities (6 decimals) and the fan-out k of
the first 400 rows of the domain's training table, so two teammates' models can be compared exactly.

    python -m pulse.experiments.model_fingerprint            # prints {"airline": "<sha>", "retail": "<sha>"}
"""

import csv
import hashlib
import json
import sys

from pulse import compat, paths

FILES = {"airline": ("action_model.pkl", "action_rows.csv", "W1"),
         "retail": ("action_model_retail.pkl", "retail_action_rows.csv", "W3")}
ROWS = 400


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fingerprint(domain: str) -> str | None:
    model_file, rows_file, wf = FILES[domain]
    mf, rf = paths.MODELS / model_file, paths.ROOT / "data" / "predictor" / rows_file
    if not mf.exists() or not rf.exists():
        return None
    m = compat.load(mf)
    h = hashlib.sha256()
    with open(rf, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= ROWS:
                break
            traj = [t for t in (r.get("last2"), r.get("last1")) if t and t not in ("START", "<s>", "NONE")]
            sig = {k: x for k, v in r.items() if k not in ("step", "task_id") and (x := num(v)) is not None}
            for key in ("gbt", "ngram"):
                p = m.proba(key, wf, traj, sig)
                h.update(json.dumps({t: round(v, 6) for t, v in sorted(p.items())}).encode())
            h.update(str(m.fanout.k("get_reservation_details" if domain == "airline" else "get_order_details", sig)).encode())
    return h.hexdigest()


if __name__ == "__main__":
    out = {d: fingerprint(d) for d in FILES}
    print(json.dumps(out))
    sys.exit(0 if all(out.values()) else 1)
