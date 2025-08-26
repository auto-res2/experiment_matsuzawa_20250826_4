"""
train.py – model construction and training routine for a tiny
Re-usable-Path Hierarchical Diffusion (RPH-Diff) proxy so that the whole
pipeline can be executed quickly on CPU yet scales to a T4 GPU when the
hyper-parameters in config/config.yaml are increased.

The implementation intentionally keeps everything in one file (model +
training code) to avoid complicated cross-file dependencies while still
respecting the relative-import rule for other modules (evaluate.py and
main.py import the model from .train).
"""
from __future__ import annotations
import os, time, random
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# ------------------------------------------------------------
#  utilities
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
#  toy 5×5 Maze environment + random dataset (proxy for D4RL)
# ------------------------------------------------------------
class TinyMazeEnv:
    SIZE = 5
    MAX_STEPS = 25
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.reset()
    def reset(self):
        self.pos = [0, 0]
        self.goal = [self.SIZE - 1, self.SIZE - 1]
        self.t = 0
        return self._obs()
    def _obs(self):
        return np.asarray(self.pos, dtype=np.float32) / (self.SIZE - 1)
    def step(self, a: int):
        dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][a]
        self.pos[0] = int(np.clip(self.pos[0] + dx, 0, self.SIZE-1))
        self.pos[1] = int(np.clip(self.pos[1] + dy, 0, self.SIZE-1))
        self.t += 1
        done = self.t >= self.MAX_STEPS or self.pos == self.goal
        rew = 1.0 if self.pos == self.goal else 0.0
        return self._obs(), rew, done, {}

class RandomMazeDataset(Dataset):
    def __init__(self, n: int = 2048):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        obs = np.random.randint(0, TinyMazeEnv.SIZE, size=2)
        goal = np.array([TinyMazeEnv.SIZE-1, TinyMazeEnv.SIZE-1])
        return {
            "obs": torch.tensor(obs / (TinyMazeEnv.SIZE-1), dtype=torch.float32),
            "goal": torch.tensor(goal / (TinyMazeEnv.SIZE-1), dtype=torch.float32),
        }

# ------------------------------------------------------------
#  model definition  (super-lightweight UNet-ish MLP)
# ------------------------------------------------------------
class SimpleTemporalUNet(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = torch.cat([x, t], dim=-1)
        return self.net(h)

class Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, h):
        return self.mlp(h)

class RPHDiff(nn.Module):
    def __init__(self,
                 shared_steps: int = 16,
                 dim: int = 64,
                 learn_taps: bool = True,
                 consistency_coef: float = 1.0):
        super().__init__()
        self.trunk = SimpleTemporalUNet(dim)
        self.h_sub = Head(dim, 2)
        self.h_act = Head(dim, 4)
        taps = torch.linspace(1, 0, shared_steps)
        self.taps = nn.Parameter(taps, requires_grad=learn_taps)
        self.cons_coef = consistency_coef

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        obs, goal = batch["obs"].to(DEVICE), batch["goal"].to(DEVICE)
        B = obs.size(0)
        t = torch.rand(B,1, device=DEVICE)
        latent = self.trunk(obs, t)
        sub = self.h_sub(latent)
        act_logits = self.h_act(latent)
        sub_loss = F.mse_loss(sub, goal)
        act_labels = torch.randint(0,4,(B,), device=DEVICE)
        act_loss = F.cross_entropy(act_logits, act_labels)
        action_vec = torch.tensor([(1,0,-1,0),(0,1,0,-1)], dtype=torch.float32, device=DEVICE)
        act_dir = F.softmax(act_logits, -1) @ action_vec.T
        cons_loss = F.mse_loss(act_dir, F.normalize(sub-obs, dim=-1))
        total = sub_loss + act_loss + self.cons_coef * cons_loss
        return {"total": total, "sub": sub_loss.detach(), "act": act_loss.detach(), "cons": cons_loss.detach()}

    @torch.no_grad()
    def sample(self, obs: np.ndarray | torch.Tensor) -> int:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(DEVICE).unsqueeze(0)
        t = torch.zeros(1,1,device=DEVICE)
        latent = self.trunk(obs, t)
        logits = self.h_act(latent)
        return int(torch.argmax(logits, -1).item())

# ------------------------------------------------------------
#  training routine
# ------------------------------------------------------------

def train_rphdiff(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    model = RPHDiff(shared_steps=cfg["model"]["shared_steps"],
                    dim=cfg["model"]["dim"],
                    learn_taps=cfg["model"]["learn_taps"],
                    consistency_coef=cfg["model"]["consistency_coef"]).to(DEVICE)

    loader = DataLoader(RandomMazeDataset(), batch_size=cfg["training"]["batch_size"], shuffle=True)
    opt = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    losses: List[float] = []
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            out["total"].backward()
            opt.step()
            losses.append(out["total"].item())
    # save training curve
    img_dir = os.path.join(".research", "iteration1", "images")
    os.makedirs(img_dir, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt; import seaborn as sns
        sns.lineplot(x=np.arange(len(losses)), y=losses)
        plt.title("Training loss (toy)")
        plt.xlabel("iter"); plt.ylabel("loss")
        plt.savefig(os.path.join(img_dir, "training_loss.pdf"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("WARNING: plotting failed –", e)
    return model
