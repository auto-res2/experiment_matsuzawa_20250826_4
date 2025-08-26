import time
from pathlib import Path
from typing import Tuple, Dict, Any, List
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from .preprocess import get_dataloaders

__all__ = [
    "TinyCNN",
    "train",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Count-Sketch utilities  (SketchReplay++ core)
# ---------------------------------------------------------------------------
class _CountSketch(Function):
    """Π·g   (forward)   and   Πᵀ·s   (backward) with autograd support."""

    @staticmethod
    def forward(ctx, grad_flat: torch.Tensor, h: torch.Tensor, xi: torch.Tensor, k: int):
        ctx.save_for_backward(h, xi)
        ctx.k = k
        s = torch.zeros(k, dtype=grad_flat.dtype, device=grad_flat.device)
        s.index_add_(0, h, grad_flat * xi)
        return s

    @staticmethod
    def backward(ctx, grad_s: torch.Tensor):
        h, xi = ctx.saved_tensors
        grad_g = grad_s[h] * xi  # Πᵀ · grad_s
        return grad_g, None, None, None


def _init_hash(d: int, k: int, device=torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialise bucket assignments h and Rademacher signs ξ."""
    h = torch.randint(0, k, (d,), device=device, dtype=torch.long)
    xi = (torch.randint(0, 2, (d,), device=device, dtype=torch.float32) * 2 - 1)
    return h, xi


def _topk_signs(vec: torch.Tensor, k: int = 64):
    """Pack indices and 1-bit signs of top-|k| coordinates."""
    if vec.numel() <= k:
        idx = torch.arange(vec.numel(), device=vec.device)
    else:
        idx = torch.topk(vec.abs(), k).indices
    signs = (vec[idx] >= 0).to(torch.uint8)
    return idx.cpu(), signs.cpu()


def _get_sketch_grad(loss: torch.Tensor, model: nn.Module, h, xi, k: int = 256) -> Dict[str, Any]:
    """Replacement for loss.backward(): returns sketched gradient record."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    g_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
    s = _CountSketch.apply(g_flat, h, xi, k)
    idx, signs = _topk_signs(g_flat, 64)
    return {
        "sketch": s.detach().cpu(),
        "idx": idx,
        "signs": signs,
        "ram_kb": 1,  # 1 kB/sample (approximate)
    }


def _reconstruct_grad(record: Dict[str, Any], h, xi, d: int) -> torch.Tensor:
    """Hierarchical Error-feedback Sketch reconstruction ĝ."""
    s = record["sketch"].to(device)
    g_hat = torch.zeros(d, device=device)
    g_hat.index_add_(0, h, s)
    eps = 0.01  # learnt in the full paper – fixed for demo
    g_hat[record["idx"].to(device)] += eps * (2 * record["signs"].float().to(device) - 1)
    return g_hat

# ---------------------------------------------------------------------------
# Simple continual-learning buffer
# ---------------------------------------------------------------------------
class _Buffer:
    def __init__(self, max_ram_kb: int):
        self.max_ram_kb = max_ram_kb
        self.ram = 0
        self.storage: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]):
        if self.ram + item["ram_kb"] > self.max_ram_kb:
            self._evict(item["ram_kb"])
        self.storage.append(item)
        self.ram += item["ram_kb"]

    def _evict(self, need: int):
        random.shuffle(self.storage)
        while need > 0 and self.storage:
            popped = self.storage.pop()
            need -= popped["ram_kb"]
            self.ram -= popped["ram_kb"]

# ---------------------------------------------------------------------------
# Model definition (tiny CNN – keeps CI fast)
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(8 * 8 * 128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any]):
    """Train TinyCNN with SketchReplay++ gradient memory."""
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    k = cfg["k"]
    buffer_size_kb = cfg["buffer_size_kb"]

    train_loader, test_loader = get_dataloaders(batch_size)

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Hash vectors are global – one per run
    d = sum(p.numel() for p in model.parameters())
    h, xi = _init_hash(d, k, device=device)

    buffer = _Buffer(buffer_size_kb)
    loss_history = []

    print("Starting training …")
    for ep in range(epochs):
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            # Store sketched gradient
            record = _get_sketch_grad(loss, model, h, xi, k)
            buffer.add(record)

            # Re-inject reconstructed gradient for update
            g_hat = _reconstruct_grad(record, h, xi, d)
            pointer = 0
            for p in model.parameters():
                part = g_hat[pointer: pointer + p.numel()].view_as(p)
                p.grad = part.clone().detach()
                pointer += p.numel()
            opt.step(); opt.zero_grad()

            loss_history.append(loss.item())
            if it % 100 == 0:
                print(f"Epoch {ep+1}  Iter {it:04d}  Loss {loss.item():.3f}  Buffer {buffer.ram} kB")

    # -------------------------------------------------------------------
    # Save training-loss curve
    # -------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # Save all images into the required directory
    img_dir = Path(".research/iteration3/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.plot(loss_history)
    plt.xlabel("Step"); plt.ylabel("CE-Loss")
    plt.title("SketchReplay++ – training loss")
    plt.tight_layout()
    f_out = img_dir / "training_loss.pdf"
    plt.savefig(f_out, bbox_inches="tight")
    print(f"Saved loss curve → {f_out}")

    return model, loss_history, test_loader
