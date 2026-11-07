"""
Download the data the project needs that is not in git (public, MIT-licensed tau-bench data).

    python -m pulse.setup.download_data              # from the repo root; skips files already there
    python -m pulse.setup.download_data --force      # download again

What it writes (gitignored):
  data/tau-bench/historical_trajectories/   tau-bench's recorded conversations, GPT-4o and Claude 3.5 Sonnet on
      {gpt-4o,sonnet-35-new}-{airline,retail}.json   airline and retail: the replays' hint signals, the lightgbm /
                                            markov training rows (build_action_data), pulse-v3's training
  data/tau-bench/schemas/{airline,retail}/  the tools' schemas (get_info) for pulse-v3's tool graph, copied from the
                                            tau-bench tool files in pulse/agent/tau_airline/tools, tau_retail/tools
Source: https://github.com/sierra-research/tau-bench (historical_trajectories/). Standard library only.
"""

import argparse
import shutil
import sys
import urllib.request

from pulse import paths

BASE = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/"
FILES = ["gpt-4o-airline.json", "sonnet-35-new-airline.json", "gpt-4o-retail.json", "sonnet-35-new-retail.json"]


def download(url, dest, force=False):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  have {dest.relative_to(paths.ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"       -> {dest.relative_to(paths.ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


def schemas():
    """pulse-v3's tool graph parses each tool's get_info() from these files (never imports them)."""
    n = 0
    for dom in ("airline", "retail"):
        out = paths.SCHEMAS / dom
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted((paths.ROOT / "pulse" / "agent" / f"tau_{dom}" / "tools").glob("*.py")):
            if f.name != "__init__.py":
                shutil.copyfile(f, out / f"{f.name}.txt")
                n += 1
    print(f"  {n} tool schemas -> {paths.SCHEMAS.relative_to(paths.ROOT)}/")


def status() -> dict:
    """What is there (the UI's Setup tab shows it)."""
    return {"trajectories": sum((paths.TRAJECTORIES / f).exists() for f in FILES), "of": len(FILES),
            "schemas": len(list(paths.SCHEMAS.glob("*/*.py.txt"))) if paths.SCHEMAS.exists() else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="download again even if the files are there")
    a = ap.parse_args()
    print("tau-bench historical trajectories:", flush=True)
    try:
        for f in FILES:
            download(BASE + f, paths.TRAJECTORIES / f, a.force)
    except OSError as e:
        sys.exit(f"download failed ({e}); check the network, or fetch {BASE}<file> by hand into {paths.TRAJECTORIES}")
    print("tool schemas:", flush=True)
    schemas()
    print("done")


if __name__ == "__main__":
    main()
