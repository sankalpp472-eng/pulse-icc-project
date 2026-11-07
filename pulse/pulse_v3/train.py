"""
Retrain pulse-v3, evaluate it against the model in use, and publish it to the S3 model registry.

  1. data      pull the training data from <uri>/_data/ (tau-bench trajectories, tool schemas, embedding cache)
  2. split     by task (seed 252): train / val (temperature) / test (held out: tasks it has never seen)
  3. train     tool graph from the training tasks' schemas + results, then the V3 model (TF-IDF of user text and
               results + graph ready set + state flags + sentence embedding)
  4. evaluate  hit@1, hit@3, warm@1..3 on the held-out tasks, for the new model AND the current latest
               (each with its own tool graph, on the same hints)
  5. gate      --promote auto: the new model becomes latest only if its warm@3 beats the current one's
               (warm@3 = share of next-step tool calls that find a warm pod with the top 3 prewarmed);
               otherwise it is published as a candidate. always / never override.
  6. publish   a new version with the metrics in its manifest; new embeddings go back to _data/

    python -m pulse.pulse_v3.train --uri s3://<bucket>/pulse-v3                     # version = today's date + time
    python -m pulse.pulse_v3.train --uri s3://<bucket>/pulse-v3 --promote never --notes "more retail data"
    python -m pulse.pulse_v3.train --local                                           # no S3: model in models/pulse_v3/

Embeddings come from the embedding server (PULSE_EMBED_BASE, llama.cpp + nomic-embed-text) for texts not in the
cache; training stops if it is needed and down (a model trained without them would not match the service).
The last line is the gate's verdict ("GATE: ..."), which the UI shows.
"""

import argparse
import json
import pickle
import time

from pulse import registry as model_store
from pulse.pulse_v3 import paths
from pulse.pulse_v3.predictors.pulse_v3 import graph  # noqa: E402
from pulse.pulse_v3.predictors.pulse_v3.embed import Embedder  # noqa: E402
from pulse.pulse_v3.predictors.pulse_v3.features import TEXT, contexts, evaluate  # noqa: E402
from pulse.pulse_v3.predictors.pulse_v3.models import Semantic  # noqa: E402
from pulse.pulse_v3.predictors.pulse_v3.predictor import load_bundle  # noqa: E402
from pulse.pulse_v3.traces import tau  # noqa: E402

GATE_METRIC = "warm@3"


def say(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--uri", help="model registry, e.g. s3://pulse-models-<account>/pulse-v3")
    ap.add_argument("--local", action="store_true", help="no S3: write the model to models/pulse_v3/<version>/")
    ap.add_argument("--version", default=time.strftime("%Y-%m-%d_%H%M"))
    ap.add_argument("--promote", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if not (a.uri or a.local):
        ap.error("give --uri (S3 registry) or --local")
    t0 = time.time()

    if a.uri:
        say(f"1/6 data: pulling {a.uri}/_data/ -> {paths.DATA}")
        say(f"    {len(model_store.data('pull', a.uri))} file(s) downloaded")
    missing = [f for f in tau.FILES if not (tau.DATA / f).exists()]
    if missing:
        raise SystemExit(f"training data missing in {tau.DATA}: {missing} (python -m pulse.setup.download_data; with S3: python -m pulse.registry data push <uri>)")

    eps = tau.load()
    train, val, test = tau.split_by_task(eps)
    say(f"2/6 split by task: {len(train)} train, {len(val)} val, {len(test)} held-out test episodes")

    say("3/6 training: tool graph, features (embeddings), V3 model")
    g = graph.build_from_tau(train)
    emb = Embedder(timeout=120)
    try:
        tr, va, te = (contexts(g, x, emb) for x in (train, val, test))
    except OSError as e:
        raise SystemExit(f"embedding server needed for {emb.new or 'some'} new texts and not reachable "
                         f"({e}); start it (llama.cpp nomic-embed-text, PULSE_EMBED_BASE)")
    model = Semantic(fields=TEXT, use_extra=True).fit(tr, va)
    new_texts = emb.new
    emb.save()

    say(f"4/6 evaluating on {len(te)} held-out hints")
    new = evaluate(model, te)
    say(f"    new:     {json.dumps(new)}")
    current, current_version = None, None
    if a.uri:
        try:
            _, _, current_version = model_store.resolve(a.uri)
            cur = load_bundle(model_store.fetch(a.uri))
            current = evaluate(cur["v3"], contexts(cur["graph"], test, emb))
            say(f"    current ({current_version}): {json.dumps(current)}")
        except Exception as e:                                # no latest yet, or it cannot be read
            say(f"    no current model to compare with ({type(e).__name__}: {e})")

    if a.promote == "always" or current is None:
        promote, why = a.promote != "never", "no current model" if current is None else "--promote always"
    elif a.promote == "never":
        promote, why = False, "--promote never"
    else:
        promote = new[GATE_METRIC] > current[GATE_METRIC]
        why = f"{GATE_METRIC} {new[GATE_METRIC]} {'>' if promote else '<='} {current[GATE_METRIC]} ({current_version})"
    say(f"5/6 gate: {'promote' if promote else 'keep as candidate'}: {why}")

    out = paths.MODELS / a.version / model_store.FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"v3": model, "graph": g, "trained_on": "tau-bench historical (GPT-4o + Sonnet 3.5)",
                     "episodes": len(train), "created": time.strftime("%Y-%m-%d %H:%M")}, f)
    metrics = {"new": new, "current": current, "current_version": current_version, "gate_metric": GATE_METRIC,
               "promoted": promote, "why": why, "split": {"train": len(train), "val": len(val), "test": len(test)}}
    if a.uri:
        say(f"6/6 publishing version {a.version}" + (" as latest" if promote else " as a candidate"))
        model_store.publish(out, a.uri, a.version, a.notes, latest=promote, metrics=metrics)
        if new_texts:
            model_store.data("push", a.uri)                   # the new embeddings, for the next training
    else:
        (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        say(f"6/6 wrote {out} (local only)")
    say(f"done in {time.time() - t0:.0f} s")
    say(f"GATE: {'promoted' if promote else 'candidate'} {a.version} ({why})")


if __name__ == "__main__":
    main()
