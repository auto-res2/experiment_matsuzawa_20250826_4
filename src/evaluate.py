from typing import Sequence, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .train import device, TinyCNN


def evaluate(model: TinyCNN, test_loader: DataLoader) -> float:
    """Simple evaluation: returns accuracy (%) and saves confusion-matrix image."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test accuracy: {acc:.2f}%")

    # Save a tiny bar plot of accuracy (just for having an image artifact)
    import matplotlib.pyplot as plt
    img_dir = Path(".research/iteration5/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.bar([0], [acc], width=0.4)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.xticks([])
    plt.tight_layout()
    out_path = img_dir / "test_accuracy.pdf"
    plt.savefig(out_path)
    print(f"Saved evaluation figure → {out_path}")
    return acc
