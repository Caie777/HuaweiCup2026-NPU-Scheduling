#!/usr/bin/env python3
"""Checkpointed, watchdog-protected official experiments for Problems 1-3.

Run from Windows or WSL. A parent process owns the global checkpoint; each
case/problem/core/algorithm unit runs in its own process and checkpoints every
official attempt. The parent can kill a blocked constructor or evaluator and
recover the last officially verified Plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_adapter import (atomic_json, evaluate_official, plan_digest,
                                replace_with_retry)
from feedback_stages import run_feedback_stages
from graph_model import GraphModel
from multi_problem_candidates import (feedback_candidates, ramp_block_candidates,
                                      route_candidates, safe_initial_plan)
from official_input import resolve_official_root
from scene_cost import score_plan

OFFICIAL = DATA = CODE = None


def load_official(value):
    """Load the official helpers supplied by the user, without bundling them."""
    global OFFICIAL, DATA, CODE
    global derive_multicore_plan, read_evaluation_config
    global read_scene_a_config, read_scene_b_config, read_cache_config
    OFFICIAL = resolve_official_root(value)
    DATA, CODE = OFFICIAL / "data", OFFICIAL / "code"
    os.environ["MATH_MODEL_OFFICIAL_ROOT"] = str(OFFICIAL)
    sys.path.insert(0, str(CODE))
    from stub_multicore_cut_and_schedule import derive_multicore_plan
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import read_scene_a_config
    from multicore_cut_evaluate_problem_2 import read_scene_b_config
    from multicore_cut_evaluate_problem_3 import read_cache_config

ALGORITHMS = ("V2plus", "RAMPplus", "OJOmacro")
PROFILES = {
    "smoke": {"max_candidates": 2, "generated": 4, "search_seconds": 8},
    "overnight": {"max_candidates": 8, "generated": 12, "search_seconds": 60},
}
_ACTIVE_WORKER = None
_STOP_REQUESTED = False


def unit_budget_seconds(args, algorithm):
    """A CLI override applies to all routes; OJO alone defaults to 420 s."""
    return (args.per_case_seconds if args.per_case_seconds is not None else
            420.0 if algorithm == "OJOmacro" else 300.0)


def official_attempt_limit(args, algorithm):
    """None means OJO has no total official-evaluation quota."""
    if algorithm == "OJOmacro" and args.max_candidates == 0:
        return None
    return args.max_candidates or PROFILES[args.profile]["max_candidates"]


def kill_process_tree(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            # Some managed Windows environments reject taskkill even for our
            # own child. Always terminate the worker itself as a fallback.
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def request_stop(_signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    kill_process_tree(_ACTIVE_WORKER)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def unit_key(case, problem, cores, algorithm):
    return f"{case}|p{problem}|c{cores}|{algorithm}"


def unit_dir(run_dir, case, problem, cores, algorithm):
    return run_dir / "runs" / case / f"p{problem}" / f"c{cores}" / algorithm


def settings():
    config = DATA / "config.txt"
    base = read_evaluation_config(config)
    return {**base, "scene_a": read_scene_a_config(config),
            "scene_b": read_scene_b_config(config),
            "cache": read_cache_config(config)}


def checkpoint_summary(args, case, problem, cores, algorithm, state,
                       *, watchdog_timeout=False, worker_exit=None,
                       interrupted=False):
    attempts = [attempt for retry in state.get("retry_history", [])
                for attempt in retry["attempts"].values()]
    attempts.extend(state.get("attempts", {}).values())
    best_hash = state.get("best_hash")
    best = state.get("attempts", {}).get(best_hash) if best_hash else None
    elapsed = state.get("total_wall_seconds")
    if elapsed is None:
        elapsed = unit_budget_seconds(args, algorithm) if watchdog_timeout else 0.0
    elapsed += sum(retry["total_wall_seconds"]
                   for retry in state.get("retry_history", []))
    exhausted = watchdog_timeout or state.get("budget_exhausted", False)
    if best:
        status = ("PARTIAL_PASS_INTERRUPTED" if interrupted else
                  "PARTIAL_PASS_BUDGET" if exhausted else
                  "PARTIAL_PASS_ERROR" if (worker_exit not in (None, 0) or
                                           state.get("generator_error") or
                                           state.get("feedback_error") or
                                           state.get("block_feedback_error"))
                  else "PASS")
    elif interrupted:
        status = "INTERRUPTED"
    elif watchdog_timeout or any(a["status"] == "TIMEOUT" for a in attempts):
        status = "TIMEOUT"
    else:
        status = "FAILED"
    return {
        "case": case, "problem": problem, "cores": cores,
        "algorithm": algorithm, "status": status,
        "profile": args.profile,
        "per_case_budget_seconds": unit_budget_seconds(args, algorithm),
        "per_eval_budget_seconds": args.per_eval_seconds,
        "candidate_limit": official_attempt_limit(args, algorithm),
        "block_extra_candidate_limit": args.ramp_block_extra_candidates,
        "total_candidate_limit": (None if official_attempt_limit(args, algorithm) is None else
                                  official_attempt_limit(args, algorithm)
                                  + (args.ramp_block_extra_candidates
                                     if algorithm == "RAMPplus" and args.ramp_block_reopt
                                     and args.feedback_policy == "route" else 0)
                                  + 1),  # Conditional safety Plan, only without a PASS.
        "feedback_policy": args.feedback_policy,
        "ramp_critical_path": args.ramp_critical_path,
        "ramp_block_reopt": args.ramp_block_reopt,
        "ramp_block_extra_candidates": args.ramp_block_extra_candidates,
        "ojo_communication_boundary": args.ojo_communication_boundary,
        "official_makespan": best.get("official_makespan") if best else None,
        "subgraph_count": best.get("subgraph_count") if best else None,
        "task_count": best.get("task_count") if best else None,
        "added_copy_bytes": best.get("added_copy_bytes") if best else None,
        "partition_added_copy_bytes": best.get("partition_added_copy_bytes") if best else None,
        "spill_added_copy_bytes": best.get("spill_added_copy_bytes") if best else None,
        "cache_hit_rate": best.get("cache_hit_rate") if best else None,
        "algorithm_seconds": (state.get("algorithm_seconds", 0.0) +
                              sum(retry["algorithm_seconds"]
                                  for retry in state.get("retry_history", []))),
        "official_evaluation_seconds": sum(a.get("official_seconds") or 0 for a in attempts),
        "official_simulator_seconds": sum(a.get("simulator_seconds") or 0 for a in attempts),
        "result_parse_seconds": sum(a.get("result_parse_seconds") or 0 for a in attempts),
        "plan_write_seconds": sum(a.get("plan_write_seconds") or 0 for a in attempts),
        "validation_seconds": state.get("validation_seconds", 0.0),
        "total_wall_seconds": elapsed,
        "phase_seconds": state.get("phase_seconds", {}),
        "current_phase": state.get("phase"),
        "constructor_diagnostics": state.get("constructor_diagnostics", {}),
        "feedback_diagnostics": state.get("feedback_diagnostics", []),
        "block_feedback_diagnostics": state.get("block_feedback_diagnostics", []),
        "feedback_stage_events": state.get("feedback_stage_events", []),
        "graph_op_count": state.get("graph_op_count"),
        "route_first": state.get("route_first", False),
        "official_attempts": len(attempts),
        "safety_fallback_official_attempts": sum(
            a.get("label") == "fresh_raw_topological_start" for a in attempts),
        "critical_path_official_attempts": sum(
            str(a.get("label", "")).startswith("critical_path_") for a in attempts),
        "block_reopt_official_attempts": sum(
            a.get("feedback_operation") == "block_reopt" for a in attempts),
        "block_reopt_official_seconds": sum(
            a.get("official_seconds") or 0 for a in attempts
            if a.get("feedback_operation") == "block_reopt"),
        "block_reopt_best": bool(best and best.get("feedback_operation") == "block_reopt"),
        "block_reopt_best_improvement": (
            best["incumbent_makespan_before"] - best["official_makespan"]
            if best and best.get("feedback_operation") == "block_reopt" and
            best.get("incumbent_makespan_before") is not None else None),
        "communication_boundary_official_attempts": sum(
            str(a.get("label", "")).endswith("communication_boundary")
            for a in attempts),
        "target_problem_official_attempts": sum(
            attempt.get("problem") == problem for attempt in attempts),
        "fixed_plan_comparison_attempts": sum(
            attempt.get("problem") != problem for attempt in attempts),
        "timeout_count": sum(a["status"] == "TIMEOUT" for a in attempts),
        "duplicate_candidate_count": state.get("duplicate_candidate_count", 0),
        "rejected_plan_count": len(state.get("rejected_plans", [])),
        "best_plan_path": best.get("plan_path") if best else None,
        "best_result_path": best.get("result_path") if best else None,
        "best_trace_path": best.get("trace_path") if best else None,
        "best_source": best.get("source") if best else None,
        "best_label": best.get("label") if best else None,
        "best_search_stage": best.get("search_stage") if best else None,
        "feedback_rounds": state.get("feedback_rounds", 0),
        "feedback_candidate_count": state.get("feedback_candidate_count", 0),
        "feedback_error": state.get("feedback_error"),
        "block_feedback_error": state.get("block_feedback_error"),
        "support_stage": ("original_scene_a_route" if problem == 1 else
                          "scene_b_constructor_only" if args.feedback_policy == "none" else
                          "scene_b_raw_op_lns" if algorithm == "OJOmacro" else
                          "scene_b_ramp_feedback" if algorithm == "RAMPplus" else
                          "stage1_scene_rerank"),
        "fixed_plan_p2_makespan": state.get("fixed_plan_p2", {}).get("official_makespan")
            if state.get("fixed_plan_p2") else None,
        "fixed_plan_p2_status": state.get("fixed_plan_p2", {}).get("status")
            if state.get("fixed_plan_p2") else None,
        "fixed_plan_p2_result_path": state.get("fixed_plan_p2", {}).get("result_path")
            if state.get("fixed_plan_p2") else None,
        "fixed_plan_cache_speedup": (
            state["fixed_plan_p2"]["official_makespan"] / best["official_makespan"]
            if problem == 3 and best and state.get("fixed_plan_p2", {}).get("status") == "PASS"
            else None),
        "generator_error": state.get("generator_error"),
        "watchdog_timeout": watchdog_timeout,
        "interrupted": interrupted,
        "worker_exit": worker_exit,
        "archived_baseline_reused": False,
        "run_id": args.run_id,
        "algorithm_version": args.version,
    }


def prepare_retry_state(state):
    """Give an unfinished unit a fresh quota while retaining its verified best."""
    attempts = state.get("attempts", {})
    best_hash = state.get("best_hash")
    best = attempts.get(best_hash)
    if best and not (Path(best["plan_path"]).is_file() and
                     Path(best["result_path"]).is_file()):
        best = None
    retained = {best_hash: best} if best else {}
    state.setdefault("retry_history", []).append({
        "attempts": {key: value for key, value in attempts.items()
                     if key not in retained},
        "total_wall_seconds": state.get("total_wall_seconds") or 0.0,
        "algorithm_seconds": state.get("algorithm_seconds", 0.0),
        "fixed_plan_p2": state.get("fixed_plan_p2"),
    })
    state["attempts"] = retained
    state["best_hash"] = best_hash if best else None
    state["algorithm_seconds"] = 0.0
    state["budget_exhausted"] = False
    state["phase"] = "starting"
    state["phase_seconds"] = {}
    state.pop("total_wall_seconds", None)
    state.pop("fixed_plan_p2", None)
    for key in ("generator_error", "feedback_error", "block_feedback_error"):
        state.pop(key, None)
    return state


def worker(args):
    case, problem, cores, algorithm = (args.worker_case, args.worker_problem,
                                       args.worker_cores, args.worker_algorithm)
    folder = unit_dir(args.output_dir, case, problem, cores, algorithm)
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "unit_checkpoint.json"
    state = (read_json(state_path, None) if not args.worker_force else None) or {
        "case": case, "problem": problem, "cores": cores, "algorithm": algorithm,
        "attempts": {}, "best_hash": None, "algorithm_seconds": 0.0,
        "budget_exhausted": False, "phase": "starting"}
    if state_path.exists() and not args.worker_force:
        state = prepare_retry_state(state)
    atomic_json(state_path, state)
    start = time.monotonic()
    deadline = start + unit_budget_seconds(args, algorithm)
    graph_path = DATA / f"{case}.json"
    state["phase"] = "graph_parse"
    atomic_json(state_path, state)
    phase_started = time.monotonic()
    raw = read_json(graph_path)
    graph = GraphModel.parse(raw)
    state.setdefault("phase_seconds", {})["graph_parse"] = time.monotonic() - phase_started
    state["graph_op_count"] = len(graph.compute_ids)
    phase_started = time.monotonic()
    cfg = settings()
    state["phase_seconds"]["config_parse"] = time.monotonic() - phase_started
    profile = PROFILES[args.profile]
    max_candidates = official_attempt_limit(args, algorithm)
    block_extra = (args.ramp_block_extra_candidates
                   if algorithm == "RAMPplus" and args.ramp_block_reopt
                   and args.feedback_policy == "route" else 0)
    total_candidates = None if max_candidates is None else max_candidates + block_extra
    generated = max(profile["generated"], max_candidates * 2) if max_candidates else profile["generated"]
    trace_policy = args.trace_policy
    state["phase"] = "initial"
    atomic_json(state_path, state)

    def evaluate(candidate, *, target_problem=problem, fixed_reference=False,
                 emergency_fallback=False, reserve_fallback=False):
        digest = f"p{target_problem}:{plan_digest(candidate['plan'])}"
        if digest in state["attempts"]:
            state["duplicate_candidate_count"] = state.get("duplicate_candidate_count", 0) + 1
            return state["attempts"][digest]
        # The independent safety Plan gets one emergency slot only when no
        # official incumbent exists. Normal candidate quotas remain unchanged.
        allowed = (None if total_candidates is None else
                   total_candidates + int(problem == 3 and fixed_reference) +
                   int(emergency_fallback and state.get("best_hash") is None))
        if time.monotonic() >= deadline or (allowed is not None and len(state["attempts"]) >= allowed):
            state["budget_exhausted"] = True
            return None
        try:
            validation_started = time.monotonic()
            view = derive_multicore_plan(raw, candidate["plan"])
            state["validation_seconds"] = state.get("validation_seconds", 0.0) + (
                time.monotonic() - validation_started)
            if view["num_cores"] != cores:
                raise ValueError("Plan core count differs from requested cores")
        except Exception as exc:
            state.setdefault("rejected_plans", []).append(
                {"label": candidate["label"], "error": f"{type(exc).__name__}: {exc}"})
            atomic_json(state_path, state)
            return None
        available = max(0.01, deadline - time.monotonic())
        if reserve_fallback and state.get("best_hash") is None:
            # Leave part of the remaining unit budget for the safety Plan if
            # this normal evaluation fails; never exceed the official cap.
            available = max(0.01, available * 0.75)
        timeout = min(args.per_eval_seconds, available)
        incumbent = state["attempts"].get(state.get("best_hash"))
        state["phase"] = f"official_evaluation:p{target_problem}:{candidate['label']}"
        atomic_json(state_path, state)
        result = evaluate_official(code_dir=CODE, graph_path=graph_path,
                                   config_path=DATA / "config.txt",
                                   plan=candidate["plan"], problem=target_problem,
                                   output_dir=folder / "candidates", timeout_seconds=timeout,
                                   trace_policy=trace_policy)
        subgraphs = len(view["subgraph_ids"])
        result.update(label=candidate["label"], source=candidate["source"],
                      search_stage=candidate.get("search_stage"),
                      scene_proxy=candidate.get("scene_proxy"),
                      route_proxy=candidate.get("route_proxy"),
                      feedback_bottleneck=candidate.get("feedback_bottleneck"),
                      feedback_operation=candidate.get("feedback_operation"),
                      block_info=candidate.get("block_info"),
                      destroy_info=candidate.get("destroy_info"),
                      incumbent_makespan_before=(incumbent.get("official_makespan")
                                                 if incumbent else None),
                      subgraph_count=subgraphs,
                      task_count=subgraphs if target_problem == 1
                      else sum(bool(row) for row in candidate["plan"]["core_schedules"]),
                      archived_baseline_reused=False)
        if incumbent and result["status"] == "PASS" and not fixed_reference:
            result["official_delta_vs_incumbent"] = (
                result["official_makespan"] - incumbent["official_makespan"])
        state["attempts"][digest] = result
        old_best_trace = None
        if result["status"] == "PASS" and not fixed_reference:
            previous = state["attempts"].get(state.get("best_hash"))
            if previous is None or result["official_makespan"] < previous["official_makespan"]:
                if previous and trace_policy == "best" and previous.get("trace_path"):
                    old_best_trace = previous["trace_path"]
                    previous["trace_path"] = None
                state["best_hash"] = digest
            elif trace_policy == "best" and result.get("trace_path"):
                Path(result["trace_path"]).unlink(missing_ok=True)
                result["trace_path"] = None
        if fixed_reference:
            state["fixed_plan_p2"] = result
        state["phase"] = "official_feedback" if not fixed_reference else "fixed_plan_p2"
        state["total_wall_seconds"] = time.monotonic() - start
        atomic_json(state_path, state)
        if old_best_trace:
            Path(old_best_trace).unlink(missing_ok=True)
        return result

    initial_started = time.monotonic()
    first = safe_initial_plan(graph, cores, cfg)
    first["scene_proxy"] = score_plan(graph, first["plan"], problem, cfg)
    state["algorithm_seconds"] += time.monotonic() - initial_started
    state["phase_seconds"]["initial_plan_construction"] = (
        time.monotonic() - initial_started)
    # This Plan is an independent output guarantee, not a route candidate.
    # Keep it ready, but evaluate the route's existing ordered list first.
    state["route_first"] = True
    if time.monotonic() < deadline and (max_candidates is None or len(state["attempts"]) < max_candidates):
        state["phase"] = "route_candidate_generation"
        atomic_json(state_path, state)
        construct_started = time.monotonic()
        diagnostics = {}
        try:
            remaining = max(1, min(profile["search_seconds"], deadline - time.monotonic()))
            reserve = (max(1, max_candidates // 3)
                       if algorithm != "V2plus" and args.feedback_policy == "route"
                       and max_candidates is not None
                       else 0)
            if algorithm == "OJOmacro" and problem == 1 and max_candidates is not None and max_candidates <= 4:
                reserve = 0
            route_limit = (generated if max_candidates is None else
                           max(1, max_candidates - 1 - reserve))
            candidates = route_candidates(graph, algorithm, problem, cores, cfg,
                                          limit=min(generated, route_limit),
                                          search_seconds=remaining,
                                          diagnostics=diagnostics,
                                          ojo_communication_boundary=(
                                              args.ojo_communication_boundary))
        except Exception as exc:
            candidates = []
            state["generator_error"] = f"{type(exc).__name__}: {exc}"
        state["algorithm_seconds"] += time.monotonic() - construct_started
        state["phase_seconds"]["route_candidate_generation"] = (
            time.monotonic() - construct_started)
        state["constructor_diagnostics"] = diagnostics
        state["phase"] = "generated"
        atomic_json(state_path, state)
        for index, candidate in enumerate(candidates):
            if time.monotonic() >= deadline or (max_candidates is not None and len(state["attempts"]) >= max_candidates):
                break
            if (index and state.get("best_hash") is None and
                    deadline - time.monotonic() <= args.per_eval_seconds):
                break
            evaluate(candidate, reserve_fallback=(
                state.get("best_hash") is None and
                plan_digest(candidate["plan"]) != plan_digest(first["plan"])))
    if state.get("best_hash") is None and time.monotonic() < deadline:
        state["phase"] = "safety_fallback"
        atomic_json(state_path, state)
        evaluate(first, emergency_fallback=True)
    def normal_feedback(limit, stage):
        best = state["attempts"][state["best_hash"]]
        state["phase"] = ("route_feedback_generation" if stage == "ordinary"
                          else "post_block_feedback_generation")
        atomic_json(state_path, state)
        feedback_started = time.monotonic()
        diagnostics = {"stage": stage, "incumbent_plan_hash": best["plan_hash"]}
        try:
            neighbors = feedback_candidates(
                graph, read_json(best["plan_path"]), read_json(best["result_path"]),
                algorithm, problem, cores, cfg,
                limit=limit,
                search_seconds=min(profile["search_seconds"],
                                   max(1, deadline - time.monotonic())),
                exclude_plan_hashes={a["plan_hash"] for a in state["attempts"].values()},
                diagnostics=diagnostics,
                ramp_critical_path=args.ramp_critical_path,
                ojo_communication_boundary=args.ojo_communication_boundary)
        except Exception as exc:
            state["feedback_error"] = f"{type(exc).__name__}: {exc}"
            neighbors = []
        state["algorithm_seconds"] += time.monotonic() - feedback_started
        phase_key = ("route_feedback_generation" if stage == "ordinary"
                     else "post_block_feedback_generation")
        state["phase_seconds"][phase_key] = (
            state["phase_seconds"].get(phase_key, 0.0)
            + time.monotonic() - feedback_started)
        state.setdefault("feedback_diagnostics", []).append(diagnostics)
        state["feedback_rounds"] = state.get("feedback_rounds", 0) + 1
        state["feedback_candidate_count"] = state.get("feedback_candidate_count", 0) + len(neighbors)
        atomic_json(state_path, state)
        return neighbors

    def block_feedback(limit):
        best = state["attempts"][state["best_hash"]]
        state["phase"] = "block_feedback_generation"
        atomic_json(state_path, state)
        started = time.monotonic()
        diagnostics = {"incumbent_plan_hash": best["plan_hash"],
                       "incumbent_makespan": best["official_makespan"]}
        try:
            neighbors = ramp_block_candidates(
                graph, read_json(best["plan_path"]), read_json(best["result_path"]),
                problem, cores, cfg, limit=limit,
                search_seconds=min(profile["search_seconds"],
                                   max(1, deadline - time.monotonic())),
                exclude_plan_hashes={a["plan_hash"] for a in state["attempts"].values()},
                diagnostics=diagnostics)
        except Exception as exc:
            state["block_feedback_error"] = f"{type(exc).__name__}: {exc}"
            neighbors = []
        elapsed = time.monotonic() - started
        state["algorithm_seconds"] += elapsed
        state["phase_seconds"]["block_feedback_generation"] = elapsed
        state.setdefault("block_feedback_diagnostics", []).append(diagnostics)
        state["block_feedback_candidate_count"] = len(neighbors)
        atomic_json(state_path, state)
        return neighbors

    if algorithm == "OJOmacro" and max_candidates is None and args.feedback_policy == "route":
        # Finite search batches and convergence remain algorithm choices;
        # there is no total official-attempt quota for the default OJO run.
        stage_events = []
        while time.monotonic() < deadline and state.get("best_hash") is not None:
            before = state["best_hash"]
            neighbors = normal_feedback(profile["generated"], "ordinary")
            stage_events.append({"stage": "ordinary", "generated": len(neighbors),
                                 "incumbent_before": before})
            for candidate in neighbors:
                if time.monotonic() >= deadline:
                    break
                evaluate(candidate)
            if not neighbors or state.get("best_hash") == before:
                break
        state["feedback_stage_events"] = stage_events
    elif algorithm != "V2plus" and args.feedback_policy == "route":
        stage_events = run_feedback_stages(
            normal_candidates=normal_feedback,
            block_candidates=(block_feedback if algorithm == "RAMPplus"
                              and args.ramp_block_reopt else None),
            evaluate=evaluate,
            best_key=lambda: state.get("best_hash"),
            used_count=lambda: len(state["attempts"]),
            has_time=lambda: time.monotonic() < deadline,
            normal_limit=max_candidates, total_limit=total_candidates)
        state["feedback_stage_events"] = stage_events
    # Same Plan under P2/P3 separates hardware cache gain from reoptimization.
    if problem == 3 and state.get("best_hash") and not state.get("fixed_plan_p2"):
        best = state["attempts"][state["best_hash"]]
        plan = read_json(best["plan_path"])
        if time.monotonic() < deadline:
            evaluate({"plan": plan, "label": "same_plan_problem2_reference",
                      "source": "fixed_plan_cache_comparison"},
                     target_problem=2, fixed_reference=True)
    state["total_wall_seconds"] = time.monotonic() - start
    state["budget_exhausted"] = state["budget_exhausted"] or time.monotonic() >= deadline
    state["phase"] = "finished"
    atomic_json(state_path, state)
    print(f"{case} p{problem} c{cores} {algorithm}: "
          f"{checkpoint_summary(args, case, problem, cores, algorithm, state)['status']} "
          f"best={checkpoint_summary(args, case, problem, cores, algorithm, state)['official_makespan']}",
          flush=True)


def normalize_cases(values):
    if len(values) == 1 and values[0].lower() == "all":
        return sorted(p.stem for p in DATA.glob("case_*.json")
                      if re.fullmatch(r"case_\d{3}", p.stem))
    return list(dict.fromkeys(v if v.startswith("case_") else f"case_{int(v):03d}"
                              for v in values))


def run_worker_process(args, case, problem, cores, algorithm):
    global _ACTIVE_WORKER
    folder = unit_dir(args.output_dir, case, problem, cores, algorithm)
    folder.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "logs" / f"{case}_p{problem}_c{cores}_{algorithm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()),
               "--official-root", str(OFFICIAL), "--_worker",
               "--worker-case", case, "--worker-problem", str(problem),
               "--worker-cores", str(cores), "--worker-algorithm", algorithm,
               "--output-dir", str(args.output_dir), "--run-id", args.run_id,
               "--profile", args.profile,
               "--per-case-seconds", str(unit_budget_seconds(args, algorithm)),
               "--per-eval-seconds", str(args.per_eval_seconds),
               "--max-candidates", str(args.max_candidates or 0),
               "--feedback-policy", args.feedback_policy,
               ("--ramp-critical-path" if args.ramp_critical_path
                else "--no-ramp-critical-path"),
               ("--ramp-block-reopt" if args.ramp_block_reopt
                else "--no-ramp-block-reopt"),
               "--ramp-block-extra-candidates", str(args.ramp_block_extra_candidates),
               ("--ojo-communication-boundary" if args.ojo_communication_boundary
                else "--no-ojo-communication-boundary"),
               "--trace-policy", args.trace_policy, "--version", args.version]
    if args.force:
        command.append("--worker-force")
    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nSTART {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=creationflags,
                                   start_new_session=(os.name != "nt"))
        _ACTIVE_WORKER = process
        watchdog = False
        try:
            process.wait(timeout=unit_budget_seconds(args, algorithm) + 2.0)
        except subprocess.TimeoutExpired:
            watchdog = True
            kill_process_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        _ACTIVE_WORKER = None
        stream.write(f"END exit={process.returncode} watchdog={watchdog} "
                     f"wall={time.monotonic()-started:.3f}\n")
    state = read_json(folder / "unit_checkpoint.json", {})
    state["total_wall_seconds"] = time.monotonic() - started
    if watchdog:
        state["budget_exhausted"] = True
    atomic_json(folder / "unit_checkpoint.json", state)
    return checkpoint_summary(args, case, problem, cores, algorithm, state,
                              watchdog_timeout=watchdog, worker_exit=process.returncode,
                              interrupted=_STOP_REQUESTED)


def write_global(args, manifest, checkpoint):
    atomic_json(args.output_dir / "manifest.json", manifest)
    atomic_json(args.output_dir / "checkpoint.json", checkpoint)
    rows = list(checkpoint["units"].values())
    atomic_json(args.output_dir / "summary.json", {"run_id": args.run_id, "rows": rows})
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temp = args.output_dir / "summary.csv.tmp"
        with temp.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        replace_with_retry(temp, args.output_dir / "summary.csv")


def git_version():
    digest = hashlib.sha256()
    sources = (sorted((ROOT / "src").glob("*.py")) +
               sorted(CODE.glob("*.py")) +
               [Path(__file__), DATA / "config.txt"])
    for path in sources:
        if path.is_relative_to(ROOT):
            label = path.relative_to(ROOT)
        elif path.is_relative_to(CODE):
            label = Path("official/code") / path.relative_to(CODE)
        else:
            label = Path("official/data") / path.name
        digest.update(label.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=3).stdout.strip() or "unversioned"
    except Exception:
        head = "unversioned"
    return f"{head}:source_sha256_{digest.hexdigest()[:16]}"


def measure_singlecore(args, cases):
    """Official no-cut single-core reference, shared by all three problems."""
    path = args.output_dir / "singlecore_baselines.json"
    saved = read_json(path, {})
    output = args.output_dir / "singlecore"
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if saved.get(case, {}).get("status") == "PASS" and not args.force:
            continue
        result_path = output / f"{case}_result.json"
        command = [sys.executable, str(CODE / "singlecore_evaluate.py"),
                   str(DATA / f"{case}.json"), "--config", str(DATA / "config.txt"),
                   "-o", str(result_path),
                   "--trace-output", str(output / f"{case}_trace.json"),
                   "--log-output", str(output / f"{case}_log.txt")]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=args.per_eval_seconds)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-2000:])
            result = read_json(result_path)
            saved[case] = {"status": "PASS", "makespan": result["makespan"],
                           "official_seconds": time.monotonic() - started,
                           "result_path": str(result_path)}
        except subprocess.TimeoutExpired as exc:
            saved[case] = {"status": "TIMEOUT", "makespan": None,
                           "official_seconds": time.monotonic() - started,
                           "error": str(exc)}
        except Exception as exc:
            saved[case] = {"status": "FAILED", "makespan": None,
                           "official_seconds": time.monotonic() - started,
                           "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(path, saved)
        print(f"SINGLECORE {case} {saved[case]['status']} "
              f"makespan={saved[case]['makespan']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path,
                        help="local contest attachment directory; alternatively set MATH_MODEL_OFFICIAL_ROOT")
    parser.add_argument("--cases", nargs="+", default=["001"])
    parser.add_argument("--cores", nargs="+", type=int, default=[2])
    parser.add_argument("--problems", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS,
                        default=list(ALGORITHMS))
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--per-case-seconds", type=float,
                        help="per-unit wall budget; default 420 s for OJOmacro, 300 s otherwise")
    parser.add_argument("--per-eval-seconds", type=float, default=120)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--feedback-policy", choices=("route", "none"),
                        default="route", help="route feedback or constructor-only ablation")
    parser.add_argument("--ramp-critical-path", action=argparse.BooleanOptionalAction,
                        default=False, help="RAMP+ timeline bottleneck feedback neighborhood")
    parser.add_argument("--ramp-block-reopt", action=argparse.BooleanOptionalAction,
                        default=False, help="RAMP+ bounded joint block reoptimization")
    parser.add_argument("--ramp-block-extra-candidates", type=int, default=0,
                        help="additional RAMP+ Block/post-Block evaluations after the normal quota")
    parser.add_argument("--ojo-communication-boundary",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="OJO raw-Op communication boundary repair")
    parser.add_argument("--trace-policy", choices=("best", "all", "none"), default="best")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-singlecore", action="store_true",
                        help="also run the official single-core no-cut reference once per case")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-problem", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-cores", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-algorithm", help=argparse.SUPPRESS)
    parser.add_argument("--worker-force", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        load_official(args.official_root)
    except ValueError as exc:
        parser.error(str(exc))
    if any(c < 1 or c > 5 for c in args.cores) or any(p not in (1, 2, 3) for p in args.problems):
        parser.error("cores must be 1..5 and problems must be 1..3")
    if ((args.per_case_seconds is not None and args.per_case_seconds <= 0) or
            args.per_eval_seconds <= 0 or
            args.max_candidates < 0 or args.ramp_block_extra_candidates < 0):
        parser.error("budgets must be positive and candidate counts non-negative")
    if args.ramp_block_extra_candidates and not args.ramp_block_reopt:
        parser.error("block extra candidates require --ramp-block-reopt")
    if args._worker:
        args.output_dir = args.output_dir.resolve()
        worker(args)
        return
    cases = normalize_cases(args.cases)
    missing = [case for case in cases if not (DATA / f"{case}.json").is_file()]
    if missing:
        parser.error(f"missing official graphs: {missing}")
    units = list(itertools.product(cases, args.problems, args.cores, args.algorithms))
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "units": len(units),
                          "problems": args.problems, "cores": args.cores,
                          "algorithms": args.algorithms,
                          "per_unit_seconds": {algorithm: unit_budget_seconds(args, algorithm)
                                               for algorithm in args.algorithms},
                          "official_attempt_limit": {algorithm: official_attempt_limit(args, algorithm)
                                                     for algorithm in args.algorithms}}, ensure_ascii=False))
        return
    args.run_id = args.run_id or time.strftime("multi_%Y%m%d_%H%M%S")
    args.output_dir = (args.output_dir or ROOT / "results" / "multi_problem" / args.run_id).resolve()
    args.version = git_version()
    existing = read_json(args.output_dir / "manifest.json")
    signature = {"profile": args.profile, "per_case_seconds": args.per_case_seconds,
                 "per_eval_seconds": args.per_eval_seconds,
                 "max_candidates": args.max_candidates,
                 "feedback_policy": args.feedback_policy,
                 "ramp_critical_path": args.ramp_critical_path,
                 "ramp_block_reopt": args.ramp_block_reopt,
                 "ramp_block_extra_candidates": args.ramp_block_extra_candidates,
                 "ojo_communication_boundary": args.ojo_communication_boundary,
                 "trace_policy": args.trace_policy}
    if existing and not (args.resume or args.force):
        parser.error("output directory already has a run; use --resume or --force")
    if existing and existing.get("signature") != signature:
        parser.error("run settings differ from manifest; choose a new output directory")
    if existing and existing.get("algorithm_version") != args.version:
        parser.error("source/config version differs from manifest; choose a new output directory")
    if existing:
        args.run_id = existing["run_id"]
        args.version = existing["algorithm_version"]
    manifest = existing or {"run_id": args.run_id,
                            "algorithm_version": args.version,
                            "signature": signature,
                            "config_path": str(DATA / "config.txt"),
                            "official_code_path": str(CODE),
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "units_requested": []}
    checkpoint = read_json(args.output_dir / "checkpoint.json", {"units": {}})
    manifest["units_requested"] = sorted(set(manifest["units_requested"]) |
                                         {unit_key(*item) for item in units})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    write_global(args, manifest, checkpoint)
    for index, (case, problem, cores, algorithm) in enumerate(units, 1):
        key = unit_key(case, problem, cores, algorithm)
        old = checkpoint["units"].get(key)
        if args.resume and not args.force and old and old["status"] in ("PASS", "PARTIAL_PASS_BUDGET"):
            print(f"SKIP {index}/{len(units)} {key} {old['status']}", flush=True)
            continue
        print(f"RUN {index}/{len(units)} {key}", flush=True)
        row = run_worker_process(args, case, problem, cores, algorithm)
        checkpoint["units"][key] = row
        write_global(args, manifest, checkpoint)
        print(f"DONE {key} {row['status']} makespan={row['official_makespan']}", flush=True)
        if _STOP_REQUESTED:
            print("STOP requested; checkpoint saved", flush=True)
            break
    if args.measure_singlecore and not _STOP_REQUESTED:
        measure_singlecore(args, cases)
    print(f"summary={args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
