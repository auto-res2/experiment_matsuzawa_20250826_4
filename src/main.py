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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/config.yaml", help="path to YAML config")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # Ensure the image output directory exists (new requirement)
    os.makedirs(".research/iteration2/images", exist_ok=True)

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
