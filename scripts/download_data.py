"""
Download the data the project needs that is not in git (public, MIT-licensed tau-bench data).

    python scripts/download_data.py              # from the repo root; skips files already there
    python scripts/download_data.py --force      # download again

What it writes (all gitignored):
  predictor/data/tau-bench/historical_trajectories/   tau-bench's recorded conversations (GPT-4o and Claude 3.5
      {gpt-4o,sonnet-35-new}-{airline,retail}.json    Sonnet, airline and retail): the replays' hint signals,
                                                       build_action_data.py (lightgbm / markov training rows)
  pulse_pred/data/tau_bench/*.json                     the same four files, where pulse-v3's training reads them
  pulse_pred/data/tau_bench/schemas/{airline,retail}/  the tools' schemas (get_info), copied from the tau-bench
                                                       tool files in agent/tau_airline/tools, agent/tau_retail/tools
Source: https://github.com/sierra-research/tau-bench (historical_trajectories/). Standard library only.
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/"
FILES = ["gpt-4o-airline.json", "sonnet-35-new-airline.json", "gpt-4o-retail.json", "sonnet-35-new-retail.json"]
TRAJ = ROOT / "predictor" / "data" / "tau-bench" / "historical_trajectories"
PULSE = ROOT / "pulse_pred" / "data" / "tau_bench"


def download(url, dest, force=False):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  have {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"       -> {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def schemas():
    """pulse-v3's tool graph parses each tool's get_info() from these files (never imports them)."""
    n = 0
    for dom in ("airline", "retail"):
        out = PULSE / "schemas" / dom
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted((ROOT / "agent" / f"tau_{dom}" / "tools").glob("*.py")):
            if f.name != "__init__.py":
                shutil.copyfile(f, out / f"{f.name}.txt")
                n += 1
    print(f"  {n} tool schemas -> {(PULSE / 'schemas').relative_to(ROOT)}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="download again even if the files are there")
    a = ap.parse_args()
    print("tau-bench historical trajectories:")
    try:
        for f in FILES:
            download(BASE + f, TRAJ / f, a.force)
    except OSError as e:
        sys.exit(f"download failed ({e}); check the network, or fetch {BASE}<file> by hand into {TRAJ}")
    print("pulse-v3 training data:")
    PULSE.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copyfile(TRAJ / f, PULSE / f)
    print(f"  4 trajectory files -> {PULSE.relative_to(ROOT)}/")
    schemas()
    print("done")


if __name__ == "__main__":
    main()
