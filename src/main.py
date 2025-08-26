"""Experiment entry-point – run with  `python -m src.main`"""
import argparse
import yaml
from pathlib import Path
from .train import train
from .evaluate import evaluate


def _resolve_default_cfg() -> Path:
    """Return the first existing default-config path.

    This utility tries a handful of sensible locations so that users do not
    have to specify --config explicitly as long as one of the default files is
    present in the repository.
    """
    candidate_paths = [
        Path("config/default.yaml"),
        Path("config/config.yaml"),  # fallback (historical name)
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError("No default configuration file found. Looked for: " + ", ".join(str(p) for p in candidate_paths))


def load_cfg(cfg_path: str | None):
    if cfg_path is None:
        path = _resolve_default_cfg()
    else:
        path = Path(cfg_path)
        if not path.exists():
            raise FileNotFoundError(f"The configuration file '{cfg_path}' does not exist.")
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
