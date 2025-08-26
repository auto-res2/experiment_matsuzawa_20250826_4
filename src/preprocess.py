"""src/preprocess.py
Light-weight preprocessing side-effects shared by training & evaluation.
Currently: sets deterministic random seeds for reproducibility.
The module is imported *for its side-effects* – do NOT remove.
"""
from __future__ import annotations

import os, random, torch, numpy as np

SEED = int(os.getenv("PYTHONHASHSEED", "1234"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
