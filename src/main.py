"""src/main.py
Driver script for the experiment.  It wires together:
  1. Training (src.train.train)
  2. Evaluation (src.evaluate.evaluate)
The script can be executed either as a module (python -m src.main) or directly.
"""
from __future__ import annotations

import argparse, pathlib, sys

from .train import train
from .evaluate import evaluate


def _parse_args():
    p = argparse.ArgumentParser(description="FlashScan-2D CIFAR-10 experiment")
    p.add_argument("--config", default="config/config.yaml", type=pathlib.Path, help="YAML configuration file")
    p.add_argument("--no-eval", action="store_true", help="Skip the evaluation step")
    return p.parse_args()


def main():
    args = _parse_args()
    ckpt_path = train(args.config)
    if not args.no_eval:
        evaluate(ckpt_path)


if __name__ == "__main__":  # pragma: no cover
    # Allow "python src/main.py" execution when the src directory is on the path
    if __package__ is None:
        # Add parent to sys.path so that imports work correctly when executed directly
        sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    main()
