"""Bounded joint core-assignment and ordering search for RAMP+ feedback.

The incumbent partition and all non-block placements stay fixed. The search
changes several existing Subgraphs at once and validates the complete quotient
DAG plus per-core serial edges before asking the route to score a full Plan.
"""
from __future__ import annotations

import heapq
import time
from collections import defaultdict

from ramp_dag import quotient_graph, sha_plan
from scene_cost import score_plan


def _acyclic_with_orders(successors, orders):
    combined = {task: set(children) for task, children in successors.items()}
    for order in orders:
        for before, after in zip(order, order[1:]):
            combined[before].add(after)
    indegree = {task: 0 for task in combined}
    for children in combined.values():
        for child in children:
            indegree[child] += 1
    ready = [task for task, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        task = heapq.heappop(ready)
        visited += 1
        for child in combined[task]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    return visited == len(combined)


def _local_order(tasks, predecessors, priority):
    remaining = set(tasks)
    indegree = {task: len(predecessors[task] & remaining) for task in remaining}
    children = defaultdict(list)
    for task in remaining:
        for parent in predecessors[task] & remaining:
            children[parent].append(task)
    ready = [(priority(task), task) for task, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        _, task = heapq.heappop(ready)
        result.append(task)
        for child in children[task]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (priority(child), child))
    return result if len(result) == len(remaining) else None


def _timeline_durations(official_result, problem):
    durations = {}
    for core in official_result.get("per_core_timeline", []):
        rows = core.get("tasks", []) if problem == 1 else core.get("subgraphs", [])
        for row in rows:
            task = row.get("subgraph_id")
            if isinstance(task, int):
                durations[task] = row.get("duration", 0)
    return durations


def _select_blocks(orders, durations, work, predecessors, successors):
    """Select two bounded two-core regions by bottleneck and dependency links."""
    if len(orders) < 2:
        return []
    loads = [sum(durations.get(task, work[task]) for task in row) for row in orders]
    heavy = max(range(len(orders)), key=lambda core: (loads[core], -core))
    if not orders[heavy]:
        return []
    linked = []
    for core in range(len(orders)):
        if core == heavy or not orders[core]:
            continue
        other = set(orders[core])
        crossing = sum(len((predecessors[task] | successors[task]) & other)
                       for task in orders[heavy])
        linked.append((-crossing, loads[core], core))
    if not linked:
        return []
    partner = min(linked)[2]

    def window(core, seed):
        row = orders[core]
        width = min(3, len(row))
        start = max(0, min(len(row) - width, row.index(seed) - width // 2))
        return tuple(row[start:start + width])

    heavy_row = orders[heavy]
    tail = heavy_row[max(0, len(heavy_row) - max(2, len(heavy_row) // 3)):]
    critical = max(tail, key=lambda task: (durations.get(task, work[task]), task))
    selected = window(heavy, critical)
    other = set(orders[partner])
    partner_seed = max(orders[partner], key=lambda task: (
        len((predecessors[task] | successors[task]) & set(selected)),
        durations.get(task, work[task]), -task))
    blocks = [(heavy, partner, selected + window(partner, partner_seed), "bottleneck")]
    # A second window probes a different part of the same pair without
    # releasing the whole graph or changing any external placement.
    if len(heavy_row) > 3 or len(orders[partner]) > 3:
        alternate = window(heavy, heavy_row[-1]) + window(partner, orders[partner][-1])
        if set(alternate) != set(blocks[0][2]):
            blocks.append((heavy, partner, alternate, "tail"))
    return blocks


def block_reopt_neighbors(route, chosen, official_result, *, problem, settings,
                          limit=3, time_budget=8.0):
    """Return legal full Plans with coupled Task/Subgraph moves and ordering.

    P1 uses independent Task order. P2/P3 use Subgraph order inside each
    core's single merged Task; scene_cost and the official evaluator retain
    the corresponding memory and cache semantics.
    """
    started = time.monotonic()
    stats = {"block_selected": 0, "block_sizes": [], "block_core_counts": [],
             "assignment_expansions": 0, "order_expansions": 0,
             "legal_candidates": 0, "duplicate_candidates": 0,
             "proxy_retained": 0, "block_seconds": 0.0}
    if limit <= 0 or route.k < 2:
        return [], stats
    groups, incumbent = chosen["groups"], chosen["plan"]
    orders = incumbent["core_schedules"]
    if len(orders) != route.k:
        return [], stats
    try:
        _, successors, predecessors, _ = quotient_graph(route.g, groups)
    except ValueError:
        return [], stats
    work = [sum(int(route.g.ops[op].get("cycles", 0)) for op in group)
            for group in groups]
    durations = _timeline_durations(official_result, problem)
    blocks = _select_blocks(orders, durations, work, predecessors, successors)
    old_core = {task: core for core, row in enumerate(orders) for task in row}
    original_position = {task: (core, at) for core, row in enumerate(orders)
                         for at, task in enumerate(row)}
    existing_hash = sha_plan(incumbent)
    candidates = {}
    deadline = started + max(0.1, time_budget)
    for core_a, core_b, block, block_kind in blocks:
        if time.monotonic() >= deadline:
            break
        stats["block_selected"] += 1
        stats["block_sizes"].append(len(block))
        stats["block_core_counts"].append(2)
        released = set(block)
        anchors = {}
        for core in (core_a, core_b):
            row = orders[core]
            first = min(i for i, task in enumerate(row) if task in released)
            anchors[core] = sum(task not in released for task in row[:first])
        base_load = [sum(work[task] for task in row) for row in orders]
        for task in block:
            base_load[old_core[task]] -= work[task]
        assignments = []
        for mask in range(1 << len(block)):
            assigned = {task: (core_b if mask & (1 << at) else core_a)
                        for at, task in enumerate(block)}
            moved = tuple(task for task in block if assigned[task] != old_core[task])
            # Include order-only exploration, but the main search explicitly
            # requires simultaneous movement of at least two Subgraphs.
            if len(moved) == 1 or len(moved) > 4:
                continue
            loads = list(base_load)
            for task, core in assigned.items():
                loads[core] += work[task]
            crossing = sum(assigned.get(parent, old_core[parent]) !=
                           assigned.get(child, old_core[child])
                           for parent in block for child in successors[parent])
            score = max(loads) + crossing * route.base.cross_wait
            exchange = any(old_core[t] == core_a for t in moved) and any(
                old_core[t] == core_b for t in moved)
            assignments.append((score, -int(exchange), abs(len(moved) - 2),
                                mask, assigned, moved))
        stats["assignment_expansions"] += len(assignments)
        assignments.sort(key=lambda row: row[:4])
        # Keep both exchanges and two-Task moves even when the cheap load
        # proxy ranks one family poorly.
        masks, seen_masks = [], set()
        for family, quota in ((lambda x: x[1] == -1, 4),
                              (lambda x: len(x[5]) >= 2 and x[1] == 0, 3),
                              (lambda x: len(x[5]) == 0, 1),
                              (lambda x: True, 2)):
            added = 0
            for row in assignments:
                if not family(row) or row[3] in seen_masks:
                    continue
                masks.append(row)
                seen_masks.add(row[3])
                added += 1
                if added >= quota:
                    break
        for _, _, _, mask, assigned, moved in masks:
            if time.monotonic() >= deadline:
                break
            for mode in ("original", "heavy_first", "reverse"):
                if time.monotonic() >= deadline:
                    break
                stats["order_expansions"] += 1
                modified = [list(row) for row in orders]
                for core in (core_a, core_b):
                    fixed = [task for task in orders[core] if task not in released]
                    members = [task for task in block if assigned[task] == core]
                    if mode == "heavy_first":
                        priority = lambda task: (-work[task], original_position[task])
                    elif mode == "reverse":
                        priority = lambda task: (-original_position[task][1],
                                                 -original_position[task][0])
                    else:
                        priority = lambda task: original_position[task]
                    local = _local_order(members, predecessors, priority)
                    if local is None:
                        break
                    at = anchors[core]
                    modified[core] = fixed[:at] + local + fixed[at:]
                else:
                    if not _acyclic_with_orders(successors, modified):
                        continue
                    plan = {"node_to_subgraph": dict(incumbent["node_to_subgraph"]),
                            "core_schedules": modified}
                    digest = sha_plan(plan)
                    if digest == existing_hash or digest in candidates:
                        stats["duplicate_candidates"] += 1
                        continue
                    try:
                        row = route._candidate(groups,
                            f"block_reopt:{block_kind}:m{len(moved)}:o{mode}:{mask}",
                            plan=plan)
                        if row is None:
                            continue
                        if problem == 1:
                            row = route._rank(row)
                            proxy = row["risk_objective"]
                        else:
                            row["scene_proxy"] = score_plan(
                                route.g, plan, problem, settings)
                            proxy = row["scene_proxy"]["objective"]
                    except (ValueError, KeyError, RuntimeError):
                        continue
                    row["feedback_operation"] = "block_reopt"
                    row["feedback_bottleneck"] = block_kind
                    row["block_info"] = {"tasks": list(block),
                                         "cores": [core_a, core_b],
                                         "moved_tasks": list(moved),
                                         "order_mode": mode,
                                         "partition_changed": False}
                    row["block_proxy"] = proxy
                    candidates[digest] = row
                    stats["legal_candidates"] += 1
                    if len(candidates) >= 18:
                        break
            if len(candidates) >= 18:
                break
    ranked = sorted(candidates.values(), key=lambda row: (
        row["block_proxy"], -len(row["block_info"]["moved_tasks"]), row["hash"]))
    # Retain assignment diversity and, when possible, a different legal order
    # for the strongest assignment. This makes ordering an actual search axis.
    def assignment_key(row):
        assignment = tuple(sorted((task, next(core for core, order in
                            enumerate(row["plan"]["core_schedules"]) if task in order))
                           for task in row["block_info"]["tasks"]))
        return assignment

    kept, masks = [], set()
    for row in ranked:
        assignment = assignment_key(row)
        if assignment in masks:
            continue
        masks.add(assignment)
        kept.append(row)
        if len(kept) >= min(limit, 2):
            break
    if len(kept) < limit and kept:
        strongest = assignment_key(kept[0])
        alternate = next((row for row in ranked
                          if assignment_key(row) == strongest and
                          row["hash"] != kept[0]["hash"]), None)
        if alternate is not None:
            kept.append(alternate)
    for row in ranked:
        if len(kept) >= limit:
            break
        if row["hash"] not in {item["hash"] for item in kept}:
            kept.append(row)
    stats["proxy_retained"] = len(kept)
    stats["block_seconds"] = time.monotonic() - started
    return kept, stats
