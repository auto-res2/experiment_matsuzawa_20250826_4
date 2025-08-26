"""src/evaluate.py
Evaluation utilities (accuracy & confusion matrix PDF).
Called by src.main after training is complete.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# relative import – keeps the package boundaries clean
from .train import _build_model  # noqa: F401 (model builder)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / ".research" / "iteration4" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(ckpt_path: pathlib.Path, batch_size: int = 256) -> float:
    """Return top-1 accuracy; also generates a confusion-matrix PDF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_val = transforms.Compose([transforms.ToTensor()])
    val_set = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=tf_val)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg_raw = checkpoint.get("cfg", {})
    # cfg was stored as a dict – wrap it for attribute-style access expected by _build_model
    cfg = cfg_raw if isinstance(cfg_raw, SimpleNamespace) else SimpleNamespace(**cfg_raw)

    num_classes = len(getattr(val_set, "classes", [])) or 10
    model = _build_model(cfg, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(labels.tolist())

    acc = sum(p == t for p, t in zip(preds, gts)) / len(gts)
    print(f"[EVAL] accuracy {acc*100:.2f}%  |  ckpt → {ckpt_path.name}")

    # Confusion matrix -------------------------------------------------------
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=False, annot=True, fmt="d")
    plt.title("CIFAR-10 confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    pdf_path = IMG_DIR / "confusion_matrix.pdf"
    plt.tight_layout(); plt.savefig(pdf_path)
    print(f"[EVAL] confusion matrix saved → {pdf_path.relative_to(PROJECT_ROOT)}")
    return acc
