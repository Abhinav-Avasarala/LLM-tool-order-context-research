"""Tiny config loader + repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_BFCL_DIR = DATA_DIR / "raw" / "bfcl"
MANIFESTS_DIR = DATA_DIR / "manifests"


def load_experiment_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
