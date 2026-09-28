"""Phase 5: run the frozen final grid against all three models, sharding
each model's run across multiple idle GPUs at once instead of one GPU
sequentially (brief section 8 doesn't mandate single-GPU serving -- this is
a wall-clock optimization, not a design change).

Why this exists (see docs/decision_log.md's 2026-09-28 entries): per-GPU
request concurrency (`run_model.py --concurrency`) does *not* meaningfully
cut wall-clock here -- H64N/H64I conditions are prefill-dominated and
already keep a single L40S compute-saturated, so batching more requests
onto the same GPU barely helps (measured: 0% for Qwen, -7% for
gpt-oss-20b). Cross-GPU sharding -- running independent server replicas on
separate idle GPUs and splitting the condition grid across them -- gives
real parallelism instead.

Shared-host discipline (explicit user requirement, not a suggestion):
never uses more than `--max-gpus` (default 5) GPUs at once, hard-capped at
6 total regardless of flags. The one case allowed to reach 6 is
gpt-oss-120b's tensor-parallel-size=2 serving: at the default cap of 5,
2 GPUs per replica only fits 2 replicas (4 GPUs) with one GPU sitting idle
and wasted, so an odd default cap is rounded up by exactly one (5 -> 6,
still <= 6) *only* for tensor_parallel_size=2 models, never beyond the
hard ceiling. An even --max-gpus, or a single-GPU model, is never bumped.

Runs one model at a time (default order: Qwen, gpt-oss-20b, gpt-oss-120b),
never multiple models' servers simultaneously -- simpler to reason about
and verify on a shared host than juggling mixed conda envs and GPU
budgets across models at once. Within a model: discovers currently-idle
GPUs, launches as many replicas as the cap and idle-GPU count allow,
shards the model's *not-yet-successfully-answered* conditions round-robin
across them, runs `run_model.py` per shard concurrently, merges, scores,
and shuts every replica down before moving on.

Safe to interrupt (Ctrl-C, SIGTERM, or a plain crash) and re-run: a
condition already sitting in the model's merged output or in a live shard
file with any status other than "inference_error" is treated as done and
never resubmitted. Retries automatically (`--max-attempts`, default 3)
within a single invocation for genuine `inference_error`s (server crash,
timeout) before giving up and raising.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from tool_context_study.config import PROJECT_ROOT

CONDA_BASE = Path("/home/aavasar/miniconda3")
HF_HOME = "/home/aavasar/hf_cache"
MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
DEFAULT_MODEL_ORDER = ["qwen3_30b_a3b_instruct_2507_fp8", "gpt_oss_20b", "gpt_oss_120b"]
HARD_GPU_CEILING = 6

FAIL_RE = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|Address already in use")

_ACTIVE_SERVERS: list[subprocess.Popen] = []


def _cleanup_all() -> None:
    for proc in list(_ACTIVE_SERVERS):
        terminate_process_group(proc)


def _signal_handler(signum: int, frame: Any) -> None:
    _cleanup_all()
    sys.exit(1)


atexit.register(_cleanup_all)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def write_status(logs_dir: Path, event: str, **fields: Any) -> None:
    """A done-signal you can check without watching stdout -- the whole point
    of running this detached is that nobody's watching a terminal. Appends
    one line to `phase5_status.jsonl` (full timeline) and overwrites
    `phase5_status.txt` (current state, `cat`-able at a glance) on every
    meaningful milestone: run/model start, replica ready, attempt progress,
    model done/failed, run done/failed."""
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "phase5_status.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    summary = " ".join(f"{k}={v}" for k, v in fields.items())
    (logs_dir / "phase5_status.txt").write_text(f"{record['time']} {event} {summary}\n")


def load_model_config(model_key: str) -> dict[str, Any]:
    with open(MODEL_CONFIG_DIR / f"{model_key}.yaml") as f:
        return yaml.safe_load(f)


def idle_gpu_indices(threshold_mib: int = 200) -> list[int]:
    """A GPU counts as idle if it's below `threshold_mib` used -- an idle L40S
    on this host reports ~17MiB (driver overhead), a loaded one is in the GBs.
    Queried fresh every time a model phase starts (not cached), since this is
    a shared host and other users' jobs come and go between phases."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    idle = []
    for line in out.strip().splitlines():
        idx_str, used_str = (x.strip() for x in line.split(","))
        if int(used_str) < threshold_mib:
            idle.append(int(idx_str))
    return idle


def effective_gpu_cap(max_gpus: int, gpus_per_replica: int) -> int:
    if max_gpus > HARD_GPU_CEILING:
        raise ValueError(f"--max-gpus may not exceed {HARD_GPU_CEILING} (shared host).")
    if gpus_per_replica == 2 and max_gpus % 2 == 1:
        return min(max_gpus + 1, HARD_GPU_CEILING)
    return max_gpus


def pick_replica_gpu_groups(cfg: dict[str, Any], max_gpus: int) -> list[list[int]]:
    gpus_per_replica = cfg.get("tensor_parallel_size", 1)
    cap = effective_gpu_cap(max_gpus, gpus_per_replica)
    idle = idle_gpu_indices()
    usable = min(cap, len(idle))
    n_replicas = usable // gpus_per_replica
    if n_replicas < 1:
        raise RuntimeError(
            f"Not enough idle GPUs: need >= {gpus_per_replica}, {len(idle)} idle "
            f"({idle}), cap {cap}."
        )
    return [idle[i * gpus_per_replica : (i + 1) * gpus_per_replica] for i in range(n_replicas)]


def terminate_process_group(proc: subprocess.Popen, grace_s: float = 15.0) -> None:
    if proc.poll() is not None:
        if proc in _ACTIVE_SERVERS:
            _ACTIVE_SERVERS.remove(proc)
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(1)
    if proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc in _ACTIVE_SERVERS:
        _ACTIVE_SERVERS.remove(proc)


def launch_server(cfg: dict[str, Any], gpu_ids: list[int], port: int, log_path: Path) -> subprocess.Popen:
    python_bin = str(CONDA_BASE / "envs" / cfg["conda_env"] / "bin" / "python")
    argv = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg["model_id"],
        "--served-model-name",
        cfg["served_model_name"],
        "--max-model-len",
        str(cfg["max_model_len"]),
        "--gpu-memory-utilization",
        str(cfg["gpu_memory_utilization"]),
        "--port",
        str(port),
    ]
    tp = cfg.get("tensor_parallel_size", 1)
    if tp > 1:
        argv += ["--tensor-parallel-size", str(tp)]
    if cfg.get("enforce_eager"):
        argv.append("--enforce-eager")
    if cfg.get("disable_custom_all_reduce"):
        argv.append("--disable-custom-all-reduce")
    if cfg.get("tool_call_parser"):
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]

    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    if tp > 1:
        env["NCCL_P2P_DISABLE"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            argv, env=env, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
        )
    _ACTIVE_SERVERS.append(proc)
    return proc


def wait_for_server_ready(base_url: str, log_path: Path, proc: subprocess.Popen, timeout_s: float) -> tuple[bool, str]:
    """Readiness is a real `GET /health` returning 200, not just a log line --
    the log-based `READY_RE` check this replaced only proved vLLM printed a
    string, not that the port was actually accepting requests. The log is
    still used for *fast failure* (a traceback or OOM message means "never
    coming up," which a health check alone can't distinguish from "still
    loading" -- both look like connection-refused)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            if FAIL_RE.search(text):
                return False, f"error pattern in log ({log_path})"
        if proc.poll() is not None:
            return False, f"process exited early (code {proc.returncode}); see {log_path}"
        try:
            resp = requests.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200:
                return True, "ready"
        except requests.RequestException:
            pass
        time.sleep(2)
    return False, f"timed out after {timeout_s}s; see {log_path}"


def launch_runner(
    cfg: dict[str, Any],
    shard_input_path: Path,
    port: int,
    output_path: Path,
    log_path: Path,
    concurrency: int | None,
) -> subprocess.Popen:
    python_bin = str(CONDA_BASE / "envs" / "tool_ctx" / "bin" / "python")
    argv = [
        python_bin,
        "-m",
        "tool_context_study.run_model",
        "--conditions",
        str(shard_input_path),
        "--base-url",
        f"http://localhost:{port}",
        "--model",
        cfg["served_model_name"],
        "--output",
        str(output_path),
    ]
    if cfg.get("api_format") == "responses":
        argv += ["--api-format", "responses"]
        if cfg.get("reasoning_effort"):
            argv += ["--reasoning-effort", cfg["reasoning_effort"]]
    if concurrency:
        argv += ["--concurrency", str(concurrency)]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        return subprocess.Popen(argv, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT)


def load_conditions_raw(path: Path) -> tuple[list[str], dict[str, str]]:
    """Keeps each condition as its original raw JSON line (not re-serialized
    through pydantic) so shard files are byte-identical to the frozen grid."""
    ordered_ids: list[str] = []
    by_id: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cid = json.loads(line)["condition_id"]
            ordered_ids.append(cid)
            by_id[cid] = line
    return ordered_ids, by_id


def shard_round_robin(items: list[str], n: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return shards


def _read_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not path.exists():
        return statuses
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = rec.get("status")
            if status is not None:
                statuses[rec["condition_id"]] = status
    return statuses


def absorb_shards_into_canonical(
    model_key: str, merged_path: Path, logs_dir: Path, ordered_ids: list[str], *, require_complete: bool
) -> dict[str, str]:
    """Folds the canonical merged file (if any) and every live shard *output*
    file into one dict keyed by condition_id (a shard's result wins over an
    older merged entry for the same id, e.g. a retried `inference_error`),
    writes it back out to `merged_path` in canonical grid order, and deletes
    the now-absorbed shard files.

    Called at the top of every attempt in `run_model_phase` -- not just once
    at the end -- so `merged_path` is the single durable source of truth for
    "done" from one attempt to the next. Earlier versions of this script only
    merged once at the very end and deleted shard files at the *start* of
    each attempt purely to avoid double-counting; that silently discarded
    already-completed results with no other attempt persisting them, causing
    completed conditions to "rotate out" and never accumulate across
    attempts within a single invocation (caught via an end-to-end resume
    smoke test, not inspection -- see docs/decision_log.md).

    With `require_complete=True`, raises if any `ordered_ids` entry is still
    missing after absorbing everything available (used for the final,
    post-loop call as a hard assertion, never mid-loop).
    """
    combined: dict[str, str] = {}
    if merged_path.exists():
        with open(merged_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    combined[json.loads(line)["condition_id"]] = line

    shard_result_paths = [
        p for p in sorted(logs_dir.glob(f"phase5_{model_key}_shard*.jsonl")) if not p.name.endswith("_input.jsonl")
    ]
    for p in shard_result_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    combined[json.loads(line)["condition_id"]] = line

    if require_complete:
        missing = [cid for cid in ordered_ids if cid not in combined]
        if missing:
            raise RuntimeError(f"[{model_key}] merge incomplete -- missing {len(missing)} condition(s): {missing[:5]}...")

    present_ordered = [cid for cid in ordered_ids if cid in combined]
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for cid in present_ordered:
            f.write(combined[cid] + "\n")
    tmp_path.replace(merged_path)

    for p in shard_result_paths:
        p.unlink()
    for p in logs_dir.glob(f"phase5_{model_key}_shard*_input.jsonl"):
        p.unlink()

    return combined


def read_done_ids(merged_path: Path) -> set[str]:
    return {cid for cid, status in _read_statuses(merged_path).items() if status != "inference_error"}


def run_score(model_key: str, merged_path: Path, final_grid_path: Path, results_dir: Path) -> None:
    python_bin = str(CONDA_BASE / "envs" / "tool_ctx" / "bin" / "python")
    summary_path = results_dir / "processed" / f"phase5_{model_key}_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [python_bin, "-m", "tool_context_study.score", "--results", str(merged_path), "--conditions", str(final_grid_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    summary_path.write_text(proc.stdout + proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith(("n_conditions", "overall_accuracy", "category_counts")):
            print(f"[{model_key}] {line}")
    print(f"[{model_key}] full score report -> {summary_path}")


def run_model_phase(
    model_key: str,
    cfg: dict[str, Any],
    ordered_ids: list[str],
    by_id: dict[str, str],
    *,
    max_gpus: int,
    results_dir: Path,
    logs_dir: Path,
    final_grid_path: Path,
    concurrency: int | None,
    max_attempts: int,
    server_timeout: float,
) -> None:
    merged_path = results_dir / "raw" / f"phase5_{model_key}.jsonl"
    write_status(logs_dir, "model_started", model=model_key)

    attempt = 0
    while True:
        attempt += 1
        # Absorbs both the existing canonical file and any shard files left
        # over from an interrupted prior attempt into `merged_path`, leaving
        # the shard dir clean -- this is what makes "done" cumulative across
        # attempts instead of resetting each time (see absorb_shards_into_canonical's
        # docstring for the bug this replaced).
        absorb_shards_into_canonical(model_key, merged_path, logs_dir, ordered_ids, require_complete=False)
        done = read_done_ids(merged_path)
        remaining_ids = [cid for cid in ordered_ids if cid not in done]
        if not remaining_ids:
            break
        if attempt > max_attempts:
            write_status(logs_dir, "model_failed", model=model_key, remaining=len(remaining_ids), attempts=max_attempts)
            raise RuntimeError(
                f"[{model_key}] {len(remaining_ids)}/{len(ordered_ids)} conditions still unresolved "
                f"after {max_attempts} attempts, e.g. {remaining_ids[:5]}"
            )
        print(f"[{model_key}] attempt {attempt}/{max_attempts}: {len(remaining_ids)}/{len(ordered_ids)} remaining")
        write_status(
            logs_dir, "attempt_started", model=model_key, attempt=attempt, remaining=len(remaining_ids), total=len(ordered_ids)
        )

        replica_gpu_groups = pick_replica_gpu_groups(cfg, max_gpus)
        n_replicas = len(replica_gpu_groups)
        print(f"[{model_key}] {n_replicas} replica(s) on GPU groups {replica_gpu_groups}")

        servers = []
        try:
            for i, gpu_ids in enumerate(replica_gpu_groups):
                port = 8000 + i
                log_path = logs_dir / f"phase5_{model_key}_server{i}.log"
                servers.append(launch_server(cfg, gpu_ids, port, log_path))

            for i, proc in enumerate(servers):
                port = 8000 + i
                log_path = logs_dir / f"phase5_{model_key}_server{i}.log"
                ok, reason = wait_for_server_ready(f"http://localhost:{port}", log_path, proc, server_timeout)
                if not ok:
                    write_status(logs_dir, "replica_failed", model=model_key, replica=i, reason=reason)
                    raise RuntimeError(f"[{model_key}] replica {i} failed to start: {reason}")
                print(f"[{model_key}] replica {i} ready (GPU(s) {replica_gpu_groups[i]}, port {port})")
                write_status(logs_dir, "replica_ready", model=model_key, replica=i, gpus=replica_gpu_groups[i], port=port)

            shards = shard_round_robin(remaining_ids, n_replicas)
            runners = []
            for i, shard_ids in enumerate(shards):
                if not shard_ids:
                    continue
                shard_input_path = logs_dir / f"phase5_{model_key}_shard{i}_input.jsonl"
                with open(shard_input_path, "w") as f:
                    for cid in shard_ids:
                        f.write(by_id[cid] + "\n")
                shard_output_path = logs_dir / f"phase5_{model_key}_shard{i}.jsonl"
                runner_log = logs_dir / f"phase5_{model_key}_runner{i}.log"
                runners.append(
                    launch_runner(cfg, shard_input_path, 8000 + i, shard_output_path, runner_log, concurrency)
                )

            for rp in runners:
                rp.wait()
            for i, rp in enumerate(runners):
                if rp.returncode != 0:
                    print(f"[{model_key}] WARNING: shard runner exited with code {rp.returncode}, see its log")
        finally:
            for proc in servers:
                terminate_process_group(proc)

    # Loop only exits once the top-of-loop absorb call found nothing missing,
    # so this is a no-op re-verification, not new work -- kept as an explicit
    # hard assertion rather than trusting the loop's own exit condition.
    absorb_shards_into_canonical(model_key, merged_path, logs_dir, ordered_ids, require_complete=True)
    print(f"[{model_key}] merged {len(ordered_ids)} results -> {merged_path}")
    run_score(model_key, merged_path, final_grid_path, results_dir)
    write_status(logs_dir, "model_done", model=model_key, n_conditions=len(ordered_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", type=Path, default=PROJECT_ROOT / "data" / "conditions" / "final_grid.jsonl")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_ORDER),
        help="Comma-separated configs/models/*.yaml stems, run in this order (one model at a time).",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=5,
        help="Cap on concurrently-used GPUs per model phase (shared host). Hard ceiling of "
        f"{HARD_GPU_CEILING} regardless -- see module docstring for the tensor-parallel rounding rule.",
    )
    parser.add_argument("--concurrency", type=int, default=None, help="Passthrough to run_model.py --concurrency per shard.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Retry rounds per model for inference_error conditions.")
    parser.add_argument("--server-timeout", type=float, default=900.0, help="Seconds to wait for each server replica to start.")
    args = parser.parse_args()

    if args.max_gpus > HARD_GPU_CEILING:
        parser.error(f"--max-gpus may not exceed {HARD_GPU_CEILING} (shared host).")

    results_dir = PROJECT_ROOT / "results"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    (results_dir / "processed").mkdir(parents=True, exist_ok=True)

    ordered_ids, by_id = load_conditions_raw(args.conditions)
    print(f"Loaded {len(ordered_ids)} conditions from {args.conditions}")
    write_status(logs_dir, "run_started", models=args.models, n_conditions=len(ordered_ids))

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    succeeded: list[str] = []
    failed: dict[str, str] = {}
    for model_key in model_keys:
        try:
            cfg = load_model_config(model_key)
            print(f"\n=== {model_key} ({cfg['served_model_name']}) ===")
            run_model_phase(
                model_key,
                cfg,
                ordered_ids,
                by_id,
                max_gpus=args.max_gpus,
                results_dir=results_dir,
                logs_dir=logs_dir,
                final_grid_path=args.conditions,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                server_timeout=args.server_timeout,
            )
            succeeded.append(model_key)
        except Exception as exc:
            # One model's failure shouldn't cost the other two -- each phase's
            # completed results are already safely merged to disk by the time
            # any exception could reach here. Logged in full (not just
            # str(exc)) since this is the one error an unattended run can't
            # ask a human to reproduce interactively.
            print(f"[{model_key}] FAILED: {exc}")
            traceback.print_exc()
            failed[model_key] = str(exc)
            write_status(logs_dir, "model_failed_uncaught", model=model_key, error=str(exc))

    print(f"\nPhase 5 baseline run finished. Succeeded: {succeeded or 'none'}. Failed: {list(failed) or 'none'}.")
    write_status(logs_dir, "run_done", succeeded=succeeded, failed=list(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
