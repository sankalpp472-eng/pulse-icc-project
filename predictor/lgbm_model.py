"""
LightGBM next-tool model: gradient-boosted decision trees over
  workflow, last tool, second-last tool, step   (+ tool-result signals, + customer intent flags)

Trained by predictor/train_lgbm.py, saved to predictor/data/lgbm.pkl, used by service.py when
PREDICTOR_MODEL=lgbm. A hint without "signals" falls back to the ensemble (see Predictor.proba).
"""

import pandas as pd

START = "START"
CATS = ["wf", "last1", "last2"]


class LGBMNextTool:
    def __init__(self, booster, features, classes, categories, feature_set):
        self.booster = booster              # lightgbm.Booster
        self.features = features            # column order the booster expects
        self.classes = list(classes)        # tool names, in the booster's class order
        self.categories = categories        # {col: [allowed values]} for the categorical columns
        self.feature_set = feature_set      # "names", "results" or "intent"

    def frame(self, rows):
        df = pd.DataFrame(rows)
        for c in CATS:
            known = df[c].where(df[c].isin(self.categories[c]))            # unknown value -> missing
            df[c] = pd.Categorical(known, categories=self.categories[c])
        for c in self.features:
            if c not in df:
                df[c] = 0
        return df[self.features]

    def proba_rows(self, rows):
        return self.booster.predict(self.frame(rows))

    def proba(self, workflow_id, trajectory, signals, vocab):
        traj = list(trajectory)
        row = {"wf": workflow_id, "last1": traj[-1] if traj else START,
               "last2": traj[-2] if len(traj) >= 2 else START, "step": len(traj) + 1, **(signals or {})}
        p = dict(zip(self.classes, self.proba_rows([row])[0]))
        out = {j: p.get(j, 0.0) + 1e-6 for j in vocab}
        s = sum(out.values())
        return {j: v / s for j, v in out.items()}

