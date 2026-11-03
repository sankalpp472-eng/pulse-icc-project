"""
Where pulse-v3 finds its files: pulse_pred/data/ (the trained bundle, the embedding cache).
PULSE_DATA_DIR and PULSE_MODELS_DIR override it.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.getenv("PULSE_DATA_DIR") or HERE / "data")
MODELS = Path(os.getenv("PULSE_MODELS_DIR") or HERE / "data")
