"""
Train the predictors' models locally from a fresh clone (no AWS). Run scripts/download_data.py first.

    python scripts/train_models.py action        # lightgbm + markov -> predictor/data/action_model.pkl   (~1 min)
    python scripts/train_models.py pulse_v3      # pulse-v3 -> pulse_pred/data/predictor.pkl              (~15-30 min)
    python scripts/train_models.py all

  action    predictor/build_action_data.py (training rows from the tau-bench airline conversations, the train /
            test split of predictor/data/steps.csv) -> predictor/train_action.py (train, score on 10 held-out tasks).
            The repo already ships a trained action_model.pkl; this reproduces it.
  pulse_v3  pulse_pred/train.py --local (tool graph + V3 model, scored on held-out tasks), then copies the model
            to pulse_pred/data/predictor.pkl, where the pulse_v3 predictor loads it without S3. Needs the
            embedding server (llama.cpp + nomic-embed-text) on PULSE_EMBED_BASE: bash scripts/embed_server.sh
The speculator has no model to train (an LLM; keys in agent/.env). With the S3 registry, train with --uri instead
(docs/RUNNING.md, section 5).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EMBED = os.getenv("PULSE_EMBED_BASE", "http://127.0.0.1:8091/v1")


def run(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)


def need(path, hint):
    if not (ROOT / path).exists():
        sys.exit(f"missing {path}: {hint}")


def action():
    need("predictor/data/tau-bench/historical_trajectories/gpt-4o-airline.json", "python scripts/download_data.py")
    run("predictor/build_action_data.py", "--tau-bench", "predictor/data/tau-bench")
    run("predictor/train_action.py")


def embed_server_up():
    try:
        with urllib.request.urlopen(EMBED.rstrip("/").removesuffix("/v1") + "/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def pulse_v3():
    need("pulse_pred/data/tau_bench/schemas/airline", "python scripts/download_data.py")
    if not embed_server_up():
        sys.exit(f"the embedding server is not reachable at {EMBED}: start it with  bash scripts/embed_server.sh  "
                 "(Docker; the first start downloads a 140 MB model), then run this again")
    version = time.strftime("%Y-%m-%d_%H%M")
    run("pulse_pred/train.py", "--local", "--version", version)
    trained = ROOT / "pulse_pred" / "data" / "models" / version / "predictor.pkl"
    shutil.copyfile(trained, ROOT / "pulse_pred" / "data" / "predictor.pkl")
    print(f"-> pulse_pred/data/predictor.pkl (the pulse_v3 predictor's default model; version {version}, "
          f"metrics in {trained.parent.relative_to(ROOT)}/metrics.json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", choices=["action", "pulse_v3", "all"])
    a = ap.parse_args()
    if a.which in ("action", "all"):
        action()
    if a.which in ("pulse_v3", "all"):
        pulse_v3()


if __name__ == "__main__":
    main()
