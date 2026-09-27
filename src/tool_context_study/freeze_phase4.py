"""Phase 4, step 5 (brief section 13): freeze configuration, manifests,
and hashes. Run only after the final catalog, the 96 approved tasks, all
final-set histories, and the full 1,152-condition grid exist and pass
validation -- this script doesn't rebuild or re-validate any of them, it
just records what's there.

Writes `data/manifests/phase4_freeze.json`: sha256 + byte count for every
frozen input file, plus the key config values inline (brief section 3:
"freeze... Generate a manifest containing file hashes").
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tool_context_study.config import CONFIG_PATH, DATA_DIR, MANIFESTS_DIR, PROJECT_ROOT, load_experiment_config

CATALOG_SPEC_PATH = PROJECT_ROOT / "configs" / "catalog_spec.yaml"
CATALOG_PATH = DATA_DIR / "catalog" / "tools.jsonl"
FINAL_TASKS_PATH = DATA_DIR / "final" / "tasks.jsonl"
FINAL_CONDITIONS_PATH = DATA_DIR / "conditions" / "final_grid.jsonl"
ASSEMBLED_DIR = DATA_DIR / "histories" / "assembled"
MODELS_DIR = PROJECT_ROOT / "configs" / "models"
SCORING_PATH = PROJECT_ROOT / "src" / "tool_context_study" / "score.py"
PARSERS_DIR = PROJECT_ROOT / "src" / "tool_context_study" / "parsers"

OUTPUT_PATH = MANIFESTS_DIR / "phase4_freeze.json"


def hash_file(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_bytes": len(data),
    }


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
        return f"{commit}{' (dirty working tree)' if dirty else ''}"
    except Exception:
        return "unknown"


def main() -> None:
    config = load_experiment_config()

    files_to_hash = [
        CATALOG_SPEC_PATH,
        CATALOG_PATH,
        FINAL_TASKS_PATH,
        FINAL_CONDITIONS_PATH,
        CONFIG_PATH,
        SCORING_PATH,
    ]
    files_to_hash += sorted(ASSEMBLED_DIR.glob("history_final_*.json"))
    files_to_hash += sorted(MODELS_DIR.glob("*.yaml"))
    files_to_hash += sorted(PARSERS_DIR.glob("*.py"))

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "frozen_design": {
            "n_tools": config["catalog"]["n_tools_total"],
            "n_final_questions": config["final_set"]["n_tools"] * config["final_set"]["n_questions_per_tool"],
            "positions": config["positions"],
            "history_conditions": config["history_conditions"],
            "history_token_targets": config["history_token_targets"],
            "interference_fraction": config["interference_fraction"],
            "final_seed": config["seeds"]["final_seed"],
            "n_final_conditions_per_model": (
                config["final_set"]["n_tools"]
                * config["final_set"]["n_questions_per_tool"]
                * len(config["history_conditions"])
                * len(config["positions"])
            ),
        },
        "files": [hash_file(p) for p in files_to_hash],
    }

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Froze {len(manifest['files'])} files -> {OUTPUT_PATH}")
    print(f"git_commit: {manifest['git_commit']}")
    print(f"n_final_conditions_per_model: {manifest['frozen_design']['n_final_conditions_per_model']}")


if __name__ == "__main__":
    main()
