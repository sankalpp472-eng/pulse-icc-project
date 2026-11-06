"""
Where pulse-v3 keeps its own files (the shared inputs are in pulse.paths: tau-bench trajectories and schemas).

  DATA     data/pulse_v3/: the sentence-embedding cache (embed_cache.npz)         PULSE_DATA_DIR overrides
  MODELS   models/pulse_v3/: predictor.pkl (the default model) and trained versions PULSE_MODELS_DIR overrides
"""

import os
from pathlib import Path

from pulse import paths as _root

DATA = Path(os.getenv("PULSE_DATA_DIR") or _root.DATA / "pulse_v3")
MODELS = Path(os.getenv("PULSE_MODELS_DIR") or _root.MODELS / "pulse_v3")
