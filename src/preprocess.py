"""src/preprocess.py
Light-weight data / seed preparation.  Currently only sets global seeds – the
CIFAR-10 dataset is fetched automatically in train.py.
The module is imported for its side-effects (deterministic behaviour).
"""
import random, numpy as np, torch, os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False  # faster whilst still reproducible
    torch.cuda.manual_seed_all(SEED)
