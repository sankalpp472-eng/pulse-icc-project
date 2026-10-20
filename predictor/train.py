"""
Step 4: train the models and choose the ensemble weights (wG, wH, wT).

How the weights are chosen (no maths needed):
  1. Hide some TRAINING tasks ("validation"), count on the rest.
  2. Try every weight combination in steps of 0.1 (wG + wH + wT = 1),
     and keep the one that is least "surprised" by the hidden tasks (lowest log-loss).
  3. Re-count on ALL training tasks with those weights.
The test tasks are never touched here; they are only used in evaluate.py.

    python predictor/train.py                    # run from the repo root

Writes predictor/data/model.pkl
"""

import argparse
import pickle
import random
from pathlib import Path

import pandas as pd

from models import (POOLED, LRModel, Predictor, WorkflowModel, examples, log_loss,
                    sequences_from_steps)

DATA = Path(__file__).resolve().parent / "data"


def weight_grid(step=0.1):
    n = round(1 / step)
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield (a * step, b * step, (n - a - b) * step)


def mean_log_loss(model, seqs, weights):
    losses = [log_loss(model.p_ensemble(traj, weights)[nxt])
              for seq in seqs for traj, nxt in examples(seq)]
    return sum(losses) / max(len(losses), 1)


def build(seqs_by_wf):
    return {wf: WorkflowModel(wf, Predictor.load_graph(wf)).fit(seqs)
            for wf, seqs in seqs_by_wf.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=252)
    a = p.parse_args()

    train = pd.read_csv(DATA / "train.csv")

    # 1. inner split of the training tasks
    tasks = sorted(train.task_id.unique())
    random.Random(a.seed).shuffle(tasks)
    val_tasks = set(tasks[:max(1, round(len(tasks) * a.val_frac))])
    fit_part = train[~train.task_id.isin(val_tasks)]
    val_part = train[train.task_id.isin(val_tasks)]
    fit_seqs, val_seqs = sequences_from_steps(fit_part), sequences_from_steps(val_part)
    tmp = build(fit_seqs)

    # 2. grid search per workflow
    best = {}
    for wf, m in sorted(tmp.items(), key=lambda kv: kv[0] != POOLED):   # pooled first
        if wf not in val_seqs:
            best[wf] = best[POOLED]     # no validation tasks for this workflow -> borrow pooled weights
            print(f"{wf:8}: no validation tasks, using the pooled weights")
            continue
        scores = {w: mean_log_loss(m, val_seqs[wf], w) for w in weight_grid()}
        best[wf] = min(scores, key=scores.get)
        print(f"{wf:8}: best weights wG={best[wf][0]:.1f} wH={best[wf][1]:.1f} wT={best[wf][2]:.1f}"
              f"   (validation log-loss {scores[best[wf]]:.3f})")

    # 3. refit on all training tasks
    all_seqs = sequences_from_steps(train)
    models = build(all_seqs)
    for wf, m in models.items():
        m.weights = best.get(wf, best.get(POOLED))
    lr = LRModel().fit(all_seqs)

    DATA.mkdir(exist_ok=True)
    with open(DATA / "model.pkl", "wb") as f:
        pickle.dump(Predictor(models, lr), f)
    print(f"-> {DATA / 'model.pkl'}")


if __name__ == "__main__":
    main()
