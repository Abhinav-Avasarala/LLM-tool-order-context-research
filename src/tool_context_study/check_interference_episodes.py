"""Automated safety checks over the interference episode bank (brief
section 6). Not a substitute for the brief's required human review pass --
see docs/decision_log.md's Phase 3 (episode banks) entry -- but catches the
mechanical failure modes a human reviewer shouldn't have to hunt for by
eye: literal tool-name leakage and answer-shaped assistant replies.

Writes one flags record per episode (empty list if clean) to
data/histories/episodes/interference_flags.json for the review tooling to
surface, and prints a summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tool_context_study.config import PROJECT_ROOT

EPISODES_DIR = PROJECT_ROOT / "data" / "histories" / "episodes"
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "tools.jsonl"
FLAGS_PATH = EPISODES_DIR / "interference_flags.json"

# Answer-shaped assistant reply heuristics -- phrasings that look like a
# tool's result being reported in prose, independent of any specific tool.
ANSWER_PATTERNS = [
    r"\bthe current (weather|temperature) (is|in .* is)\b",
    r"\bthe capital (city )?of \w+ is\b",
    r"\bthe population of \w+ is\b",
    r"\bthe (distance|driving distance) (is|between)\b",
    r"\byour monthly payment (is|would be|comes out to)\b",
    r"\bthe (resistance|capacitance|magnetic field|electric field) is\b",
    r"\bthe (exchange rate|converted amount) is\b",
    r"\b\d+(\.\d+)?\s*(ohms|farads|amps|volts|newtons)\b",
]


def load_catalog_names() -> list[tuple[str, str]]:
    names = []
    with open(CATALOG_PATH) as f:
        for line in f:
            d = json.loads(line)
            names.append((d["name"], d["original_name"]))
    return names


def check_episode(ep: dict, catalog_names: list[tuple[str, str]]) -> list[str]:
    flags = []
    full_text = " ".join(m["content"] for m in ep["messages"]).lower()

    for normalized, original in catalog_names:
        candidates = {normalized.lower(), original.lower(), normalized.lower().replace("__", ".")}
        for cand in candidates:
            if cand in full_text:
                flags.append(f"literal_tool_name:{normalized}")
                break

    for pattern in ANSWER_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            flags.append(f"answer_shaped_reply:{pattern}")

    return flags


def main() -> None:
    catalog_names = load_catalog_names()
    all_results = {}
    n_flagged = 0
    n_total = 0

    for path in sorted(EPISODES_DIR.glob("interference_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                n_total += 1
                flags = check_episode(ep, catalog_names)
                all_results[ep["episode_id"]] = flags
                if flags:
                    n_flagged += 1

    with open(FLAGS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Checked {n_total} interference episodes.")
    print(f"Flagged: {n_flagged}")
    if n_flagged:
        for episode_id, flags in all_results.items():
            if flags:
                print(f"  {episode_id}: {flags}")
    print(f"Wrote {FLAGS_PATH}")


if __name__ == "__main__":
    main()
