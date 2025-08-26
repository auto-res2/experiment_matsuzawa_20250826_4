"""Experiment entry-point – run with  `python -m src.main`"""
import argparse
import yaml
from pathlib import Path
from .train import train
from .evaluate import evaluate


def load_cfg(cfg_path: str):
    dflt_path = Path("config/default.yaml")
    path = Path(cfg_path) if cfg_path else dflt_path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model, _, test_loader = train(cfg["train"])
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
