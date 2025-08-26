"""src/train.py
Training routine for the FlashScan-2D experiments.
The script is *never* executed directly – it is imported by src.main so please
keep side-effects to a minimum.
"""
from __future__ import annotations

import pathlib, time, yaml, random, os
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import preprocess  # noqa: F401 (ensures preprocess side-effects such as seed)

# ---------------------------------------------------------------------------
# Model helpers – FlashScan-2D is optional, we fall back to ResNet18 when absent
# ---------------------------------------------------------------------------
try:
    import flashscan2d  # type: ignore
    from flashscan2d import FlashVSS
    FLASH_OK = True
except (ImportError, AttributeError):
    FLASH_OK = False

import timm  # tiny memory footprint + many models ready-made

# ---------------------------------------------------------------------------
# I/O utils – save **everything** below .research/iteration4/images
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / ".research" / "iteration4" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train(cfg_path: pathlib.Path | str = "config/config.yaml") -> pathlib.Path:
    """Main entry called by src.main.  Returns the checkpoint path."""
    cfg = _load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data ----------------------------------------------------------------
    train_loader, val_loader = _build_dataloaders(cfg)

    # Derive the number of classes from the dataset (works for CIFAR-10 & friends)
    num_classes = len(getattr(train_loader.dataset, "classes", [])) or 10  # fallback to 10

    # 2) Model ---------------------------------------------------------------
    model = _build_model(cfg, num_classes).to(device)

    # 3) Optimiser / AMP -----------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler() if device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss()

    # 4) Training loop -------------------------------------------------------
    best_acc, ckpt_path = 0.0, MODEL_DIR / "flashscan_cifar.pt"
    for epoch in range(1, cfg.epochs + 1):
        model.train(); epoch_loss, n, t0 = 0.0, 0, time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
            epoch_loss += loss.item() * labels.size(0); n += labels.size(0)
        epoch_loss /= n

        # 5) Validation ------------------------------------------------------
        acc = _validate(model, val_loader, device)
        print(f"[TRAIN] epoch {epoch:02d}/{cfg.epochs} | loss {epoch_loss:.4f} | val-acc {acc*100:5.2f}% | {(time.time()-t0):.1f}s")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    print(f"[TRAIN] Finished – best accuracy {best_acc*100:.2f}%  |  model saved → {ckpt_path}")
    return ckpt_path

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_cfg(path: pathlib.Path | str) -> SimpleNamespace:
    with open(path) as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def _build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([transforms.ToTensor()])

    root = pathlib.Path("data/cifar10")
    root.mkdir(parents=True, exist_ok=True)

    train_set = datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
    val_set = datasets.CIFAR10(root, train=False, download=True, transform=tf_val)

    nw = getattr(cfg, "num_workers", 0)  # safer default inside containers
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def _build_model(cfg, num_classes: int):
    if FLASH_OK:
        base = timm.create_model("vmamba_small", pretrained=False, num_classes=num_classes)
        model = FlashVSS.from_vmamba(base, scan_version="flash2d", recompute=getattr(cfg, "recompute", False))
    else:
        model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    return model


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            pred = logits.argmax(1).cpu()
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)
