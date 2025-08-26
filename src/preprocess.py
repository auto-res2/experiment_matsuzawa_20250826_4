"""
preprocess.py – Placeholder preprocessing stage.
In a larger project you would download datasets or perform expensive
feature generation here.  For this tiny example we only log the call so
that the rest of the pipeline remains intact.
"""
from __future__ import annotations
from typing import Dict, Any


def run_preprocessing(cfg: Dict[str, Any]):
    """No-op preprocessing stage (required by the project template)."""
    if cfg.get("preprocessing", {}).get("verbose", False):
        print("[preprocess] – preprocessing stage skipped (nothing to do).")
