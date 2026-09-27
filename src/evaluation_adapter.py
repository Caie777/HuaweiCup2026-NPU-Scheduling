"""Problem 1/2/3 official CLI evaluation without changing official code."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:20]


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    replace_with_retry(temp, path)


def replace_with_retry(source, target):
    """Allow brief Windows reader locks while preserving atomic replacement."""
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(min(0.01 * (2 ** attempt), 0.2))


def evaluate_official(*, code_dir, graph_path, config_path, plan, problem,
                      output_dir, timeout_seconds, trace_policy="best"):
    """Persist the exact Plan and call the selected official evaluator once."""
    if problem not in (1, 2, 3):
        raise ValueError("problem must be 1, 2, or 3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = plan_digest(plan)
    prefix = f"p{problem}_{digest}"
    plan_path = output_dir / f"{prefix}_plan.json"
    result_path = output_dir / f"{prefix}_result.json"
    trace_path = output_dir / f"{prefix}_trace.json"
    log_path = output_dir / f"{prefix}_log.txt"
    plan_write_started = time.monotonic()
    atomic_json(plan_path, plan)
    plan_write_seconds = time.monotonic() - plan_write_started
    command = [sys.executable, str(Path(code_dir) / f"multicore_cut_evaluate_problem_{problem}.py"),
               str(graph_path), str(plan_path), "--config", str(config_path),
               "-o", str(result_path), "--trace-output", str(trace_path),
               "--log-output", str(log_path)]
    start = time.monotonic()
    row = {"status": "FAILED", "problem": problem, "plan_hash": digest,
           "plan_path": str(plan_path), "result_path": None,
           "trace_path": None, "log_path": str(log_path),
           "official_seconds": None, "official_makespan": None,
           "added_copy_bytes": None, "partition_added_copy_bytes": None,
           "spill_added_copy_bytes": None, "cache_hit_rate": None,
           "cache_stats": None, "error": None}
    row["plan_write_seconds"] = plan_write_seconds
    row["simulator_seconds"] = None
    row["result_parse_seconds"] = None
    try:
        simulator_started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=max(0.01, float(timeout_seconds)))
        row["simulator_seconds"] = time.monotonic() - simulator_started
        if completed.returncode:
            raise RuntimeError(f"official exit {completed.returncode}: {completed.stderr[-2000:]}")
        parse_started = time.monotonic()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row["result_parse_seconds"] = time.monotonic() - parse_started
        movement = result.get("data_movement_bytes", {})
        cache = result.get("cache_stats") if problem == 3 else None
        row.update(status="PASS", official_makespan=result["makespan"],
                   result_path=str(result_path), trace_path=str(trace_path),
                   added_copy_bytes=movement.get("added_copy_bytes"),
                   partition_added_copy_bytes=movement.get("partition_added_copy_bytes"),
                   spill_added_copy_bytes=movement.get("spill_added_copy_bytes"),
                   cache_hit_rate=cache.get("hit_rate") if cache else None,
                   cache_stats=cache)
    except subprocess.TimeoutExpired as exc:
        row["simulator_seconds"] = time.monotonic() - simulator_started
        row.update(status="TIMEOUT", error=str(exc))
    except Exception as exc:
        row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    row["official_seconds"] = time.monotonic() - start
    if row["status"] != "PASS":
        result_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)
    elif trace_policy == "none":
        trace_path.unlink(missing_ok=True)
        row["trace_path"] = None
    return row
