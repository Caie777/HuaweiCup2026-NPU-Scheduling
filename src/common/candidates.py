"""Shared fallback construction and scene-aware candidate ranking."""
from __future__ import annotations
import json
import math
from common.schedule import make_plan
from common.scene_cost import score_plan


def safe_initial_plan(graph, cores, settings):
    """A bounded raw-topological start, independent of archived solutions."""
    order = graph.compute_order
    count = min(len(order), max(1, 2 * cores))
    width = max(1, math.ceil(len(order) / count))
    groups = [order[i:i + width] for i in range(0, len(order), width)]
    plan, _ = make_plan(graph, groups, cores, bandwidth=settings["bandwidth"],
                        cross_wait=settings["scene_a"]["task_cross_core_wait_cycles"],
                        same_wait=settings["scene_a"]["task_same_core_wait_cycles"])
    return {"plan": plan, "label": "fresh_raw_topological_start",
            "groups": groups, "source": "fresh", "search_stage": "initial"}


def _rank(graph, rows, problem, settings, limit):
    scored = []
    for row in rows:
        try:
            proxy = score_plan(graph, row["plan"], problem, settings)
            scored.append({**row, "scene_proxy": proxy})
        except (KeyError, ValueError):
            continue
    scored.sort(key=lambda row: (row["scene_proxy"]["objective"],
                                 len(set(row["plan"]["node_to_subgraph"].values())),
                                 row["label"]))
    chosen, seen, sizes = [], set(), set()
    for row in scored:
        digest = json.dumps(row["plan"], sort_keys=True, separators=(",", ":"))
        if digest in seen:
            continue
        task_count = len(set(row["plan"]["node_to_subgraph"].values()))
        if task_count not in sizes or len(chosen) >= max(1, limit // 2):
            chosen.append(row)
            seen.add(digest)
            sizes.add(task_count)
        if len(chosen) >= limit:
            break
    return chosen
