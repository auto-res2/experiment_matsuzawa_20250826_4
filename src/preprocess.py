"""
preprocess.py – trivial because we use a synthetic dataset.
Placed here mainly for completeness and to respect the required file
layout.  In a real experiment this module would download / clean data
sets, create train/val/test splits, cache processed artefacts, …
"""
from typing import Any, Dict

from torch.utils.data import DataLoader

from .train import RandomMazeDataset

def get_dataloader(batch_size: int = 128) -> DataLoader:
    ds = RandomMazeDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# placeholder for potential future preprocessing steps

def run_preprocessing(cfg: Dict[str, Any]):
    print("No preprocessing necessary for the toy example – skipping.")
