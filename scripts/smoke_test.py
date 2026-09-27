#!/usr/bin/env python3
"""Small, attachment-free check of all three routes and budget reporting."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate import validate_partition
from aggregate_v2 import FastV1ModuleAggregator
from aggregate_v2plus import V2PlusAggregator
from graph_model import GraphModel
from ojo_lns import OJOLNS
from ramp_dag import quotient_graph
from ramp_plus import RAMPPlus
from run_all_problems import checkpoint_summary, official_attempt_limit, unit_budget_seconds


def synthetic_graph():
    edges = ((0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5))
    ops = [{"id": 100 + i, "op": "CONV", "pipe": "PIPE_M" if i % 2 else "PIPE_V",
            "cycles": 20 + i * 4} for i in range(6)]
    tensors, links = [], []
    for i, (source, target) in enumerate(edges):
        tensor = 1000 + i
        tensors.append({"id": tensor, "pos": "UB", "size": 128})
        links += [{"source": ops[source]["id"], "target": tensor},
                  {"source": tensor, "target": ops[target]["id"]}]
    for i in (0, 1):
        tensor = 1100 + i
        tensors.append({"id": tensor, "pos": "DDR", "size": 256})
        links.append({"source": tensor, "target": ops[i]["id"]})
    tensors.append({"id": 1200, "pos": "DDR", "size": 256})
    links.append({"source": ops[5]["id"], "target": 1200})
    return GraphModel.parse({"ops": ops, "tensors": tensors, "edges": links})


def check_candidate(graph, candidate):
    groups, plan = candidate["groups"], candidate["plan"]
    validate_partition(graph, groups)
    quotient_graph(graph, groups)
    assert sorted(op for group in groups for op in group) == graph.compute_ids
    assert sorted(task for core in plan["core_schedules"] for task in core) == list(range(len(groups)))


def main():
    graph = synthetic_graph()
    common = {"bandwidth": 60, "same_wait": 100, "cross_wait": 200,
              "capacity": {"L1": 4096, "UB": 4096}}
    modules = FastV1ModuleAggregator(graph, cache_bytes=8192).run()
    groups, v2 = V2PlusAggregator(graph, modules, 2, **common).run()
    check_candidate(graph, {"groups": groups, "plan": v2["selected"]["plan"]})
    ramp = RAMPPlus(graph, 2, **common).initial_candidates(time_budget=2, limit=3)
    assert ramp["candidates"]
    check_candidate(graph, ramp["candidates"][0])
    ojo = OJOLNS(graph, 2, **common).solve(iterations=2, seed_count=4,
                                          time_budget=2, multiscale=True, macro=True)
    assert ojo["candidates"]
    check_candidate(graph, ojo["candidates"][0])

    args = SimpleNamespace(per_case_seconds=None, per_eval_seconds=120,
                           max_candidates=0, profile="overnight",
                           ramp_block_extra_candidates=0, ramp_block_reopt=False,
                           feedback_policy="route", ramp_critical_path=False,
                           ojo_communication_boundary=False, run_id="synthetic",
                           version="synthetic")
    assert unit_budget_seconds(args, "OJOmacro") == 420
    assert official_attempt_limit(args, "OJOmacro") is None
    assert unit_budget_seconds(args, "V2plus") == 300
    assert official_attempt_limit(args, "RAMPplus") == 8
    saved = {"status": "PASS", "official_makespan": 123, "problem": 1,
             "plan_path": "already_verified_plan.json", "result_path": "verified_result.json"}
    state = {"attempts": {"best": saved}, "best_hash": "best", "budget_exhausted": True}
    row = checkpoint_summary(args, "case_001", 1, 2, "OJOmacro", state,
                             watchdog_timeout=True)
    assert row["status"] == "PARTIAL_PASS_BUDGET"
    assert row["official_makespan"] == 123
    assert row["best_plan_path"] == saved["plan_path"]
    assert row["candidate_limit"] is None
    print("SMOKE OK: V2plus, RAMPplus, OJOmacro; 420 s / unlimited OJO quota; verified Plan survives timeout")


if __name__ == "__main__":
    main()
