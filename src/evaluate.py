"""
evaluate.py – evaluation helpers for the RPH-Diff toy experiments.
"""
from __future__ import annotations
import time, json, random
from typing import Dict, Any

import numpy as np
import torch

from .train import TinyMazeEnv, DEVICE

# ------------------------------------------------------------
#  evaluation – success-rate + latency + VRAM
# ------------------------------------------------------------

def gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()/(1024**3)
    return 0.0

def evaluate_policy(model, episodes: int = 100, env_seed: int = 0) -> Dict[str, Any]:
    env = TinyMazeEnv(seed=env_seed)
    successes, latencies = 0, []
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter_ns()
            action = model.sample(obs)
            t1 = time.perf_counter_ns()
            latencies.append((t1-t0)/1e6)
            obs, rew, done, _ = env.step(action)
        successes += int(rew>0)
    res = {
        "success_rate": successes/episodes,
        "latency_ms": np.mean(latencies),
        "vram_gb": gpu_mem_gb(),
    }
    print(json.dumps(res, indent=2))
    return res
