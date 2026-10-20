"""
Step 7: load test of the prediction API (experiment E3, requirement NFR-4: p95 < 50 ms at 10 hints/s).

Sends hints at a FIXED rate (1, 5, 10, 20 per second by default), taken from real test
trajectories, and measures each answer's latency on the client side.
Start the API first (service.py), in another terminal.

    python predictor/loadtest.py                               # run from the repo root
    python predictor/loadtest.py --rates 10 --duration 60 --url http://127.0.0.1:8000
"""

import argparse
import asyncio
import csv
import random
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

from models import examples, sequences_from_steps

HERE = Path(__file__).resolve().parent


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def make_hints(seed):
    seqs = sequences_from_steps(pd.read_csv(HERE / "data" / "test.csv"))
    seqs.pop("airline", None)
    hints = [{"workflow_id": wf, "episode_id": f"load-{i}", "trajectory": traj}
             for wf, ss in seqs.items() for i, seq in enumerate(ss) for traj, _ in examples(seq)]
    random.Random(seed).shuffle(hints)
    return hints


async def run_rate(client, url, hints, rate, duration):
    lat, errors = [], 0

    async def one(h):
        nonlocal errors
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=h)
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors += 1

    tasks, start, n = [], time.perf_counter(), int(rate * duration)
    for i in range(n):
        # open loop: send at the planned time, whether or not earlier answers came back
        await asyncio.sleep(max(0.0, start + i / rate - time.perf_counter()))
        tasks.append(asyncio.create_task(one(hints[i % len(hints)])))
    await asyncio.gather(*tasks)
    return {"rate_per_s": rate, "sent": n, "errors": errors,
            "p50_ms": statistics.median(lat), "p95_ms": pct(lat, 95), "p99_ms": pct(lat, 99),
            "max_ms": max(lat)}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--rates", type=float, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--duration", type=float, default=30, help="seconds per rate")
    p.add_argument("--seed", type=int, default=252)
    a = p.parse_args()

    hints = make_hints(a.seed)
    rows = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        await client.post(a.url + "/predict", json=hints[0])          # warm-up
        for rate in a.rates:
            print(f"rate {rate:>4} hints/s for {a.duration:.0f} s ...", flush=True)
            r = await run_rate(client, a.url + "/predict", hints, rate, a.duration)
            ok = "PASS" if r["p95_ms"] < 50 and r["errors"] == 0 else "FAIL"
            print(f"   p50 {r['p50_ms']:.1f} ms  p95 {r['p95_ms']:.1f} ms  p99 {r['p99_ms']:.1f} ms"
                  f"  errors {r['errors']}   [{ok}: p95 < 50 ms]")
            rows.append(r)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"loadtest_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
