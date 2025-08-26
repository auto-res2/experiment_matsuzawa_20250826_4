"""Data-loading utilities.

For CI / offline environments we avoid external downloads by using torchvision's
`FakeData` dataset. It produces random images on-the-fly, which is perfectly
sufficient for ensuring that the training loop executes without errors.
"""
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

__all__ = ["get_dataloaders"]

def get_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Return train & test dataloaders using fake data.

    Parameters
    ----------
    batch_size: int
        Batch size for the training dataloader.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    transform = transforms.ToTensor()

    train_ds = FakeData(size=1024,  # small for fast CI runs
                        image_size=(3, 32, 32),
                        num_classes=10,
                        transform=transform)

    test_ds = FakeData(size=256,
                       image_size=(3, 32, 32),
                       num_classes=10,
                       transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    return train_loader, test_loader
