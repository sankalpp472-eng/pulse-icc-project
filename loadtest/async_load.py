"""
Open-loop load with asyncio + httpx: requests go out on a fixed schedule whether or not earlier ones answered
(Locust's users are closed-loop: each waits for its answer). The right shape for an SLO check, and for exactly
repeatable cold-start storms.

  predictor  hints at exactly R per second for each rate (sessions kept in order: one stream per rate unit)
               python loadtest/async_load.py predictor --url http://127.0.0.1:8000 --rates 1 5 10 20 --duration 60
  tools      bursts: N concurrent calls to each chosen read-only tool at the same instant, repeated
               python loadtest/async_load.py tools --burst 1 5 10 --pause 90
             (--pause > the tools' idle window (60 s) + scale-down makes every burst start cold)

Writes the same run folder as the Locust tests (common.Recorder): results/load_runs/<date>_load-<target>_asyncio_...
"""

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

import common  # noqa: E402


async def predictor(a, rec: common.Recorder):
    eps = common.episodes(a.seed, common.parse_tasks(a.tasks))
    async with httpx.AsyncClient(base_url=a.url, timeout=30) as client:
        for rate in a.rates:
            streams = [common.hint_stream(eps, f"r{rate}-{i}") for i in range(max(1, int(rate)))]
            n, t0, jobs = int(rate * a.duration), time.time() + 0.2, []

            async def one(h, rate=rate):
                t = time.time()
                try:
                    r = await client.post("/predict", json=h)
                    ok, err = r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"
                except Exception as e:
                    ok, err = False, type(e).__name__
                rec.add(t, rate, "hint", "hint", time.time() - t, ok, err)

            for k in range(n):                                  # open loop: send on schedule, never wait
                await asyncio.sleep(max(0.0, t0 + k / rate - time.time()))
                jobs.append(asyncio.create_task(one(next(streams[k % len(streams)]))))
            await asyncio.gather(*jobs)
            print(f"  {rate}/s: {n} hints sent", flush=True)


async def tools(a, rec: common.Recorder):
    calls = common.read_only_calls(common.episodes(a.seed, common.parse_tasks(a.tasks)))
    names = a.tools or sorted({t for t, _ in calls})
    rng = random.Random(a.seed)
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(tool, args, burst):
            url, headers, body = common.mcp_request(tool, args)
            t = time.time()
            try:
                r = await client.post(url, json=body, headers=headers)
                err = common.mcp_failed(r.status_code, r.text)
            except Exception as e:
                err = type(e).__name__
            rec.add(t, burst, "tool", f"tool:{tool}", time.time() - t, not err, err)

        for burst in a.burst:
            for rep in range(a.repeat):
                jobs = [one(t, rng.choice([g for n, g in calls if n == t]), burst) for t in names for _ in range(burst)]
                t0 = time.time()
                await asyncio.gather(*jobs)
                print(f"  burst {burst} x {len(names)} tools (#{rep + 1}): {time.time() - t0:.1f} s", flush=True)
                if a.pause:
                    await asyncio.sleep(a.pause)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", choices=["predictor", "tools"])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="predictor: its address")
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 5, 10, 20], help="predictor: hints per second")
    ap.add_argument("--duration", type=float, default=60, help="predictor: seconds per rate")
    ap.add_argument("--burst", type=int, nargs="+", default=[1, 5, 10], help="tools: concurrent calls per tool")
    ap.add_argument("--repeat", type=int, default=1, help="tools: bursts per size")
    ap.add_argument("--pause", type=float, default=0, help="tools: seconds between bursts (90+ = cold every time)")
    ap.add_argument("--tools", nargs="+", help="tools: which read-only tools (default: every one in the traces)")
    ap.add_argument("--config", default="C0", help="C0 / C1 / C3: recorded with the results")
    ap.add_argument("--seed", type=int, default=252)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    params = {k: v for k, v in vars(a).items() if k not in ("target", "label")}
    rec = common.Recorder(a.target, "asyncio", a.config, params)
    out = common.run_dir(a.target, "asyncio", a.config, a.label)
    try:
        asyncio.run(predictor(a, rec) if a.target == "predictor" else tools(a, rec))
    finally:
        if rec.rows:
            rec.write(out)


if __name__ == "__main__":
    main()
