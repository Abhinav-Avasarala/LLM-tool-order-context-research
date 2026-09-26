"""Download and read the BFCL files this project reuses (brief section 4).

Pulls from the Hugging Face dataset mirror at a pinned revision (see
`configs/experiment.yaml: bfcl.dataset_revision`), copies them into
`data/raw/bfcl/` for a project-local, hash-checked copy, and exposes
readers that parse the JSONL files into `RawBFCLQuestion` / `RawBFCLAnswer`.

A provenance manifest recording the exact revision, retrieval time, and
per-file sha256 is written to `data/manifests/bfcl_provenance.json` on every
download, so any later catalog/task built from this data can be traced back
to exactly what was pulled and when.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from tool_context_study.config import MANIFESTS_DIR, RAW_BFCL_DIR, load_experiment_config
from tool_context_study.schemas import RawBFCLAnswer, RawBFCLQuestion


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_bfcl_files(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Download every file in `bfcl.files` at the pinned revision.

    Returns a mapping of {relative_filename: local_path_under_data/raw/bfcl}.
    Writes/overwrites the provenance manifest as a side effect.
    """
    config = config or load_experiment_config()
    bfcl_cfg = config["bfcl"]
    repo_id = bfcl_cfg["dataset_repo"]
    revision = bfcl_cfg["dataset_revision"]
    repo_type = bfcl_cfg.get("repo_type", "dataset")

    RAW_BFCL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    local_paths: dict[str, Path] = {}
    manifest_files: list[dict[str, Any]] = []

    for filename in bfcl_cfg["files"]:
        cached_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=filename,
            )
        )
        dest_path = RAW_BFCL_DIR / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, dest_path)

        local_paths[filename] = dest_path
        manifest_files.append(
            {
                "filename": filename,
                "sha256": _sha256_file(dest_path),
                "n_bytes": dest_path.stat().st_size,
            }
        )

    manifest = {
        "dataset_repo": repo_id,
        "dataset_revision": revision,
        "repo_type": repo_type,
        "github_repo": bfcl_cfg.get("github_repo"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    }
    with open(MANIFESTS_DIR / "bfcl_provenance.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return local_paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_questions(filename: str, revision: str) -> list[RawBFCLQuestion]:
    path = RAW_BFCL_DIR / filename
    return [
        RawBFCLQuestion(
            id=rec["id"],
            question=rec["question"],
            function=rec["function"],
            source_file=filename,
            source_revision=revision,
        )
        for rec in _read_jsonl(path)
    ]


def load_answers(filename: str, revision: str) -> list[RawBFCLAnswer]:
    path = RAW_BFCL_DIR / filename
    return [
        RawBFCLAnswer(
            id=rec["id"],
            ground_truth=rec["ground_truth"],
            source_file=filename,
            source_revision=revision,
        )
        for rec in _read_jsonl(path)
    ]


def load_questions_with_answers(
    question_filename: str,
    answer_filename: str,
    revision: str,
) -> list[tuple[RawBFCLQuestion, RawBFCLAnswer]]:
    """Join a question file with its `possible_answer/` counterpart by id.

    BFCL's own `live_*` files occasionally contain a question id with no
    matching ground-truth id upstream (observed: 1/1053 in
    BFCL_v3_live_multiple.json, an apparent id typo in BFCL itself, e.g.
    `...-79-0` vs `...-279-0`). This is development-set scaffolding, not
    this project's frozen final set, so such rows are skipped with a
    logged warning rather than failing ingestion outright.
    """
    questions = load_questions(question_filename, revision)
    answers_by_id = {a.id: a for a in load_answers(answer_filename, revision)}

    joined = []
    skipped: list[str] = []
    for q in questions:
        answer = answers_by_id.get(q.id)
        if answer is None:
            skipped.append(q.id)
            continue
        joined.append((q, answer))

    if skipped:
        print(
            f"Warning: {len(skipped)} question(s) in {question_filename} have no "
            f"matching ground truth in {answer_filename}, skipped: {skipped}"
        )
    return joined


def main() -> None:
    config = load_experiment_config()
    local_paths = download_bfcl_files(config)
    revision = config["bfcl"]["dataset_revision"]

    print(f"Downloaded {len(local_paths)} files at revision {revision}:")
    for filename, path in local_paths.items():
        print(f"  {filename} -> {path}")

    multiple = load_questions_with_answers(
        "BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json", revision
    )
    live_multiple = load_questions_with_answers(
        "BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json", revision
    )
    print(f"BFCL_v3_multiple: {len(multiple)} question/answer pairs")
    print(f"BFCL_v3_live_multiple: {len(live_multiple)} question/answer pairs")


if __name__ == "__main__":
    main()
