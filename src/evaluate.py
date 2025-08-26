from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

__all__ = ["evaluate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model: torch.nn.Module, loader: DataLoader) -> Dict[str, Any]:
    """Return accuracy on the provided DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}%")
    return {"accuracy": acc}
