"""
evaluate.py – simple evaluation utilities for the TinyMaze experiment.
The evaluator runs a fixed number of episodes in the TinyMazeEnv and
returns a dictionary that will be written to experiment_metrics.json by
main.py.
"""
from __future__ import annotations
from typing import Dict, Any

import numpy as np

# Import TinyMazeEnv and RPHDiff from the training module to avoid code
# duplication.  We deliberately use relative import to satisfy the
# enforced import rules.
from .train import TinyMazeEnv, RPHDiff


def evaluate_policy(model: RPHDiff, episodes: int = 20) -> Dict[str, Any]:
    """Evaluate *model* in TinyMazeEnv for *episodes* and return metrics."""
    env = TinyMazeEnv()
    success = 0
    steps_total = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        rew = 0.0
        # Roll-out until either the agent reaches the goal or MAX_STEPS.
        while not done:
            action = model.sample(obs)
            obs, rew, done, _ = env.step(action)
            steps += 1
        success += 1 if rew > 0 else 0
        steps_total += steps

    return {
        "episodes": episodes,
        "success_rate": success / episodes,
        "avg_steps": steps_total / episodes,
    }
