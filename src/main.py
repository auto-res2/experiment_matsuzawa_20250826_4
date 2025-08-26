"""
main.py – experiment entry point.  Run from project root via

    python -m src.main

This script wires together preprocessing, training and evaluation using
the YAML file stored under config/config.yaml.
"""
from __future__ import annotations
import argparse, os, yaml, json

from .preprocess import run_preprocessing
from .train import train_rphdiff
from .evaluate import evaluate_policy


def load_cfg(path: str) -> dict:
    """Load YAML config; fall back to default values when the file is empty."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return DEFAULT_CFG.copy()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Merge with defaults so that missing keys are silently filled in.
    merged = DEFAULT_CFG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict):
            merged.setdefault(k, {}).update(v)
        else:
            merged[k] = v
    return merged


DEFAULT_CFG = {
    "model": {
        "shared_steps": 16,
        "dim": 64,
        "learn_taps": True,
        "consistency_coef": 1.0,
    },
    "training": {
        "batch_size": 128,
        "lr": 5e-4,
        "epochs": 3,
        "seed": 42,
    },
    "evaluation": {
        "episodes": 20,
    },
    "preprocessing": {
        "verbose": False,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/config.yaml", help="path to YAML config")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # Ensure the image output directory exists (new requirement)
    os.makedirs(".research/iteration3/images", exist_ok=True)

    # 1) (optional) preprocessing stage
    run_preprocessing(cfg)

    # 2) training
    model = train_rphdiff(cfg)

    # 3) evaluation
    results = evaluate_policy(model, episodes=cfg["evaluation"]["episodes"])

    # 4) save metrics for later aggregation
    with open("experiment_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinished.  Detailed metrics stored in experiment_metrics.json")


if __name__ == "__main__":  # pragma: no cover
    main()
