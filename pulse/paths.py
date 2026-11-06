"""
Where everything lives. Code finds files through these names, never through its own location.

  ROOT        the repository (in the runner image: /app)
  DATA        inputs: data/predictor (training rows), data/traces (replay traces), data/coldstart (measured cold
              starts for the simulated cluster), data/tau-bench (downloaded, see pulse.setup.download_data)
  MODELS      trained models: models/action_model.pkl (lightgbm + markov, in git), models/pulse_v3/ (trained or
              downloaded), models/registry/ (versions fetched from the S3 registry)
  RESULTS     experiment and load-test output (gitignored)
  ENV_FILE    .env: LLM keys and settings (gitignored; .env.example shows what goes in)

PULSE_DATA and PULSE_MODELS override them (pods and Jobs, whose image folders are read-only). Experiment runs
write to RESULTS/local_runs or RESULTS/cloud_runs unless PULSE_RESULTS_DIR names the folder (run pods do).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("PULSE_DATA") or ROOT / "data")
MODELS = Path(os.getenv("PULSE_MODELS") or ROOT / "models")
RESULTS = ROOT / "results"
ENV_FILE = ROOT / ".env"
INFRA = ROOT / "infra"

PREDICTOR_DATA = DATA / "predictor"
TRACES = DATA / "traces"
COLDSTART = DATA / "coldstart"
TAU_BENCH = DATA / "tau-bench"
TRAJECTORIES = TAU_BENCH / "historical_trajectories"
SCHEMAS = TAU_BENCH / "schemas"
REGISTRY_CACHE = Path(os.getenv("PULSE_REGISTRY_CACHE") or MODELS / "registry")
