"""
Train the predictors' models locally from a fresh clone (no AWS). Run pulse.setup.download_data first.

    python -m pulse.setup.train_models action        # lightgbm + markov -> models/action_model.pkl     (~1 min)
    python -m pulse.setup.train_models action_retail # the same for retail -> models/action_model_retail.pkl
    python -m pulse.setup.train_models pulse_v3      # pulse-v3 -> models/pulse_v3/predictor.pkl        (~45 min)
    python -m pulse.setup.train_models all

  action    build_action_data (training rows from the tau-bench airline conversations, the train / test tasks of
            data/predictor/steps.csv) -> train_action (train, score on the 10 held-out tasks). The repository already
            ships a trained models/action_model.pkl; this reproduces it.
  action_retail  the same from the tau-bench retail conversations (the LLM speculator's split: 92 training tasks,
            23 held out, listed in data/predictor/retail_split.json); the retail replays' lightgbm / markov use it.
  pulse_v3  pulse.pulse_v3.train --local (tool graph + V3 model, scored on held-out tasks), then copies the model to
            models/pulse_v3/predictor.pkl, where the pulse_v3 predictor loads it without S3. Needs the embedding
            server (llama.cpp + nomic-embed-text) on PULSE_EMBED_BASE: bash scripts/embed_server.sh
The speculator has no model to train (an LLM; keys in .env). With the S3 registry, train with --uri instead
(the UI's Models tab).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from pulse import paths
from pulse.pulse_v3 import paths as v3_paths

PY = sys.executable
EMBED = os.getenv("PULSE_EMBED_BASE", "http://127.0.0.1:8091/v1")


def run(module, *args):
    print("$ python -m", module, *args, flush=True)
    subprocess.run([PY, "-m", module, *args], cwd=paths.ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": str(paths.ROOT)})


def need(path, hint):
    if not path.exists():
        sys.exit(f"missing {path.relative_to(paths.ROOT)}: {hint}")


def action():
    need(paths.TRAJECTORIES / "gpt-4o-airline.json", "python -m pulse.setup.download_data")
    run("pulse.predictor.build_action_data")
    run("pulse.predictor.train_action")


def action_retail():
    need(paths.TRAJECTORIES / "gpt-4o-retail.json", "python -m pulse.setup.download_data")
    run("pulse.predictor.build_action_data", "--domain", "retail")
    run("pulse.predictor.train_action", "--domain", "retail")


def embed_server_up():
    try:
        with urllib.request.urlopen(EMBED.rstrip("/").removesuffix("/v1") + "/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def pulse_v3():
    need(paths.SCHEMAS / "airline", "python -m pulse.setup.download_data")
    if not embed_server_up():
        sys.exit(f"the embedding server is not reachable at {EMBED}: start it with  bash scripts/embed_server.sh  "
                 "(Docker; the first start downloads a 140 MB model), then run this again")
    version = time.strftime("%Y-%m-%d_%H%M")
    run("pulse.pulse_v3.train", "--local", "--version", version)
    trained = v3_paths.MODELS / version / "predictor.pkl"
    shutil.copyfile(trained, v3_paths.MODELS / "predictor.pkl")
    print(f"-> {(v3_paths.MODELS / 'predictor.pkl').relative_to(paths.ROOT)} (the pulse_v3 predictor's default model; "
          f"version {version}, metrics in {trained.parent.relative_to(paths.ROOT)}/metrics.json)")


def status() -> dict:
    """Which models are there (the UI's Setup tab shows it)."""
    return {"action": (paths.MODELS / "action_model.pkl").exists(),
            "action_retail": (paths.MODELS / "action_model_retail.pkl").exists(),
            "pulse_v3": (v3_paths.MODELS / "predictor.pkl").exists(), "embed_server": embed_server_up()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", choices=["action", "action_retail", "pulse_v3", "all"])
    a = ap.parse_args()
    if a.which in ("action", "all"):
        action()
    if a.which in ("action_retail", "all"):
        action_retail()
    if a.which in ("pulse_v3", "all"):
        pulse_v3()


if __name__ == "__main__":
    main()
