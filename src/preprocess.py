"""src/preprocess.py
Light-weight data / seed preparation.  Currently only sets global seeds – the
CIFAR-10 dataset is fetched automatically in train.py.
The module is imported for its side-effects (deterministic behaviour).
"""
import os, random

import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    # Using benchmark=False can speed things up while still remaining deterministic
    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed_all(SEED)
