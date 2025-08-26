from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset


class _RandomImageDataset(Dataset):
    """Small synthetic dataset of random images and labels (10 classes)."""
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.images = torch.rand(n_samples, 3, 32, 32)
        self.labels = torch.randint(0, 10, (n_samples,))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders using synthetic data.

    Using a synthetic dataset avoids external downloads and keeps CI fast.
    """
    train_ds = _RandomImageDataset(1024)
    test_ds = _RandomImageDataset(256)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
