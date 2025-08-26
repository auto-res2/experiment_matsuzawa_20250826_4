from typing import Tuple
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

__all__ = ["get_dataloaders"]

def get_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return DataLoaders."""
    tf = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
