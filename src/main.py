"""src/main.py
Entry-point invoked via `python -m src.main`.  Handles
   1) optional preprocessing (seeds / data already in place)
   2) training
   3) evaluation
The script prints concise progress information to stdout so that the user can
track the entire pipeline.
"""
from __future__ import annotations

import argparse, pathlib, sys

from . import preprocess  # noqa: F401  (import for side-effects)
from .train import train
from .evaluate import evaluate

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse_args():
    p = argparse.ArgumentParser(description="FlashScan-2D CIFAR-10 demo pipeline")
    p.add_argument("--cfg", default="config/config.yaml", help="Path to YAML config file")
    p.add_argument("--skip-train", action="store_true", help="Only run evaluation – expects existing model")
    return p.parse_args()


def main():
    args = _parse_args()

    cfg_path = pathlib.Path(args.cfg)
    if not cfg_path.exists():
        print(f"[MAIN] Config {cfg_path} not found – aborting."); sys.exit(1)

    ckpt_path = PROJECT_ROOT / "models" / "flashscan_cifar.pt"

    if not args.skip_train:
        print("[MAIN] ---------- TRAINING ----------")
        ckpt_path = train(cfg_path)
    else:
        if not ckpt_path.exists():
            print("[MAIN] checkpoint missing and --skip-train given – aborting."); sys.exit(1)
        print("[MAIN] skipping training – using existing", ckpt_path.name)

    print("[MAIN] ---------- EVALUATION ----------")
    accuracy = evaluate(ckpt_path)
    print(f"[MAIN] DONE – final accuracy {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
