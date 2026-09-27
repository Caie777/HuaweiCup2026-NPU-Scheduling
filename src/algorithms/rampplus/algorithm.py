"""RAMP+: schedule-feedback refinement with Step1-aware Task risk.

The frozen RAMPDAG remains unchanged. This class adds global balanced seeds,
calibrated risk ranking, and legal split/merge/move/reorder neighbors that can
be evaluated after official feedback.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from common.partition import validate_partition
from common.memory import estimate_task, estimate_task_step1, index_compute_edges
from algorithms.rampplus.dag import PROFILES, RAMPDAG, _Coarsener, sha_plan
from common.partition import quotient_graph


class RAMPPlus:
    def __init__(self, graph, cores, *, bandwidth, same_wait, cross_wait, capacity):
        self.base = RAMPDAG(graph, cores, bandwidth=bandwidth,
                            same_wait=same_wait, cross_wait=cross_wait,
                            capacity=capacity)
        self.g, self.k = graph, cores
        self.bandwidth, self.capacity = bandwidth, dict(capacity)
        self._risk_cache = {}
        self._step1_edge_index = index_compute_edges(graph)

    def _risk(self, members):
        key = tuple(sorted(members))
        if key not in self._risk_cache:
            try:
                risk = estimate_task_step1(self.g, members, self.capacity,
                                           edge_index=self._step1_edge_index)
                source = "official_step1_induced_graph"
            except Exception as exc:
                risk = estimate_task(self.g, members, self.capacity)
                source = f"topological_fallback:{type(exc).__name__}"
            self._risk_cache[key] = (risk, source)
        return self._risk_cache[key]

    def _rank(self, row):
        potential_spill = 0; risky = []
        for task, group in enumerate(row["groups"]):
            risk, source = self._risk(group)
            overflow = sum(risk["overflow_bytes"].values())
            multiplier = min(12.0, 2.0 + len(group) / 256.0)
            potential_spill += overflow * multiplier
            if overflow:
                risky.append({"task": task, "overflow_bytes": overflow,
                              "risk_source": source,
                              "peak_live_bytes": risk["peak_live_bytes"],
                              "high_risk_tensors": risk.get("high_risk_tensors", [])[:4]})
        estimate = row["estimate"]
        previous_spill = estimate.get("spill_bytes_proxy", 0)
        adjusted = estimate["objective_proxy"] - previous_spill / self.bandwidth
        adjusted += potential_spill / self.bandwidth
        row = dict(row)
        row["risk_objective"] = adjusted
        row["potential_spill_bytes"] = potential_spill
        row["risky_tasks"] = sorted(risky, key=lambda r: (-r["overflow_bytes"], r["task"]))
        return row

    def _candidate(self, groups, label, decision=None, plan=None):
        validate_partition(self.g, groups)
        if plan is None:
            plan, estimate = self.base._schedule(groups)
        else:
            estimate = self.base._evaluate_fixed_schedule(groups, plan)
            if estimate is None:
                return None
        return {"groups": [list(x) for x in groups], "plan": plan,
                "estimate": estimate, "label": label,
                "decisions": [decision] if decision else [],
                "hash": sha_plan(plan), "task_count": len(groups)}

    def initial_candidates(self, *, time_budget=45, limit=12):
        started = time.monotonic()
        solved = self.base.solve(time_budget=time_budget, top_k=max(8, limit))
        base_elapsed = time.monotonic() - started
        candidates = {row["hash"]: row for row in solved["top_candidates"]}
        coarse = solved.get("best_initial_coarse")
        if coarse:
            candidates.setdefault(coarse["hash"], coarse)
        # Global workload-balanced seeds are generated from the same natural
        # modules, even when local coarsening stops before reaching p Tasks.
        phase_started = time.monotonic()
        coarsener = _Coarsener(self.g, self.base.modules, "balanced", PROFILES["balanced"],
                              self.bandwidth, self.capacity, self.k)
        m = len(self.base.modules)
        counts = sorted({m, min(m, self.k), min(m, 2*self.k),
                         min(m, 4*self.k), min(m, 8*self.k)})
        seen_group_modes = set()
        duplicate_global_seeds = 0
        for count in counts:
            module_groups, reason = self.base._rebalance_modules(coarsener, count)
            groups = [[op for mid in mids for op in self.base.modules[mid]]
                      for mids in module_groups]
            for mode in ("heft", "topological"):
                signature = (tuple(tuple(group) for group in groups), mode)
                if signature in seen_group_modes:
                    duplicate_global_seeds += 1
                    continue
                seen_group_modes.add(signature)
                try:
                    plan, estimate = self.base._schedule(groups, mode=mode)
                    row = {"groups": groups, "plan": plan, "estimate": estimate,
                           "label": f"global_balance_{count}:{reason}:{mode}",
                           "decisions": [{"operation": "global_balance_seed",
                                          "requested_tasks": count, "reason": reason}],
                           "hash": sha_plan(plan), "task_count": len(groups)}
                    candidates.setdefault(row["hash"], row)
                except Exception:
                    continue
        global_seed_elapsed = time.monotonic() - phase_started
        phase_started = time.monotonic()
        ranked = [self._rank(row) for row in candidates.values()]
        ranked.sort(key=lambda r: (r["risk_objective"],
                                   r["estimate"]["estimated_makespan_proxy"],
                                   r["task_count"], r["hash"]))
        return {"candidates": ranked[:limit], "all_ranked_count": len(ranked),
                "duplicate_global_seeds_skipped": duplicate_global_seeds,
                "frozen_ramp_candidate_count": solved["candidate_count"],
                "frozen_ramp_candidate_errors": len(solved["all_candidate_errors"]),
                "frozen_ramp_search_timed_out": solved["timed_out_during_search"],
                "phase_seconds": {**solved["phase_seconds"],
                                  "base_solve_total": base_elapsed,
                                  "global_seed_scheduling": global_seed_elapsed,
                                  "risk_proxy_ranking": time.monotonic() - phase_started},
                "algorithm_seconds": time.monotonic() - started}

    def _split_task(self, groups, task_id):
        members = set(groups[task_id])
        if len(members) < 2:
            return None
        mids = sorted({self.base.module_of[op] for op in members},
                      key=lambda mid: self.base.module_order_pos[mid])
        if len(mids) >= 2:
            pieces = [list(self.base.modules[mid]) for mid in mids]
            weights = [sum(self.g.ops[op].get("cycles", 0) for op in group) for group in pieces]
            total = sum(weights); prefix = 0
            best = None
            for i, weight in enumerate(weights[:-1], 1):
                prefix += weight
                candidate = (abs(total - 2*prefix), i)
                if best is None or candidate < best:
                    best = candidate
            cut = best[1]
            left = [op for piece in pieces[:cut] for op in piece]
            right = [op for piece in pieces[cut:] for op in piece]
            kind = "split_natural_modules"
        else:
            local = [op for op in self.g.compute_order if op in members]
            total = sum(self.g.ops[op].get("cycles", 0) for op in local)
            prefix = 0; best = None
            for i, op in enumerate(local[:-1], 1):
                prefix += self.g.ops[op].get("cycles", 0)
                candidate = (abs(total - 2*prefix), i)
                if best is None or candidate < best:
                    best = candidate
            cut = best[1]
            left, right = local[:cut], local[cut:]
            kind = "split_op_topological"
        proposal = [list(g) for g in groups]
        proposal[task_id:task_id+1] = [left, right]
        try:
            validate_partition(self.g, proposal)
        except Exception:
            return None
        return proposal, kind

    def _split_task_fraction_fixed(self, groups, plan, task_id, fraction):
        """Split one subgraph at a different topological workload quantile.

        Both pieces stay in the same merged-core Task and preserve its order.
        This probes Step1 residency rather than claiming a same-core COPY gain.
        """
        members = set(groups[task_id])
        order = [op for op in self.g.compute_order if op in members]
        if len(order) < 2:
            return None
        total = sum(self.g.ops[op].get("cycles", 0) for op in order)
        target = total * fraction
        work = 0
        cut = 1
        for index, op in enumerate(order[:-1], 1):
            work += self.g.ops[op].get("cycles", 0)
            cut = index
            if work >= target:
                break
        pieces = [list(g) for g in groups]
        pieces[task_id:task_id + 1] = [order[:cut], order[cut:]]
        try:
            validate_partition(self.g, pieces)
        except ValueError:
            return None
        left_ops = set(pieces[task_id])
        schedules = []
        for row in plan["core_schedules"]:
            replaced = []
            for old in row:
                if old == task_id:
                    replaced.extend((task_id, task_id + 1))
                else:
                    replaced.append(old + (old > task_id))
            schedules.append(replaced)
        new_plan = {"node_to_subgraph": {
            op: task + (task > task_id) if task != task_id else
            (task_id if int(op) in left_ops else task_id + 1)
            for op, task in plan["node_to_subgraph"].items()},
            "core_schedules": schedules}
        return pieces, new_plan

    def _merge_targets(self, groups):
        owner = {op: task for task, group in enumerate(groups) for op in group}
        _, succ, _, _ = quotient_graph(self.g, groups)
        savings = defaultdict(int)
        for tid, row in self.base.tensor_views.items():
            producers = {owner[op] for op in row["producers"]}
            consumers = {owner[op] for op in row["consumers"]}
            for a in producers:
                for b in consumers:
                    if a != b and (b in succ[a] or a in succ[b]):
                        savings[tuple(sorted((a,b)))] += row["bytes"]
        return sorted(savings, key=lambda pair: (-savings[pair], pair))[:3]

    def block_reopt_neighbors(self, chosen, official_result, *, problem,
                              settings, limit=3, time_budget=8.0):
        """Jointly reschedule a bounded two-core block of existing Subgraphs."""
        from algorithms.rampplus.block import block_reopt_neighbors
        return block_reopt_neighbors(
            self, chosen, official_result, problem=problem, settings=settings,
            limit=limit, time_budget=time_budget)

    def critical_path_neighbors(self, chosen, official_result, *, problem,
                                settings=None, limit=3):
        """Target a timeline-derived bottleneck chain, not an exact critical path.

        P1 timeline entries are Tasks. P2/P3 timeline entries below the merged
        core Task are Subgraphs, so the two scenes use separate target logic.
        """
        if limit <= 0:
            return []
        groups, plan = chosen["groups"], chosen["plan"]
        orders = plan["core_schedules"]
        core_of = {task: core for core, row in enumerate(orders) for task in row}
        timeline = official_result.get("per_core_timeline", [])
        if len(timeline) != self.k or not groups:
            return []
        targets = []
        if problem == 1:
            times = {}
            for core in timeline:
                for item in core.get("tasks", []):
                    tid = item.get("subgraph_id", item.get("task_id"))
                    if isinstance(tid, int) and 0 <= tid < len(groups):
                        times[tid] = item
            if not times:
                return []
            predecessor = defaultdict(set)
            for edge in official_result.get("task_dependencies", []):
                source, target = edge.get("source"), edge.get("target")
                if source in times and target in times:
                    predecessor[target].add(source)
            for row in orders:
                for source, target in zip(row, row[1:]):
                    predecessor[target].add(source)
            tail = max(times, key=lambda tid: (times[tid].get("end", 0),
                                               times[tid].get("duration", 0), -tid))
            chain, seen = [], set()
            while tail not in seen:
                chain.append(tail)
                seen.add(tail)
                prior = predecessor[tail] - seen
                if not prior:
                    break
                tail = max(prior, key=lambda tid: (times[tid].get("end", 0), -tid))
            targets = sorted(chain, key=lambda tid: (
                -times[tid].get("duration", 0), -times[tid].get("end", 0), tid))[:2]
        elif problem in (2, 3):
            finish = [max((item.get("end", 0) for item in core.get("tasks", [])),
                          default=0) for core in timeline]
            heavy = max(range(self.k), key=lambda core: (finish[core], -core))
            subgraphs = timeline[heavy].get("subgraphs", [])
            targets = [row["subgraph_id"] for row in sorted(
                subgraphs, key=lambda row: (-row.get("duration", 0),
                                            -row.get("start", 0),
                                            row.get("subgraph_id", -1)))
                if row.get("subgraph_id") in core_of][:2]
        else:
            raise ValueError(problem)
        proposals = {}

        def add(candidate, operation):
            if candidate and candidate["hash"] not in proposals:
                candidate["feedback_operation"] = f"critical_path_{operation}"
                candidate["feedback_bottleneck"] = "timeline_backtrace_proxy"
                proposals[candidate["hash"]] = candidate

        def try_add(candidate_groups, label, operation, fixed_plan=None):
            try:
                add(self._candidate(candidate_groups, label, plan=fixed_plan),
                    operation)
            except (ValueError, KeyError, RuntimeError):
                pass

        for task in targets:
            source = core_of[task]
            for target in range(self.k):
                if target == source:
                    continue
                for position in sorted({0, len(orders[target]) // 2,
                                        len(orders[target])}):
                    modified = [list(row) for row in orders]
                    modified[source].remove(task)
                    modified[target].insert(position, task)
                    fixed = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                             "core_schedules": modified}
                    try_add(groups,
                            f"critical_path_move_{task}_{source}_{target}_{position}",
                            "move", fixed)
            at = orders[source].index(task)
            for other in (at - 1, at + 1):
                if 0 <= other < len(orders[source]):
                    modified = [list(row) for row in orders]
                    modified[source][at], modified[source][other] = (
                        modified[source][other], modified[source][at])
                    fixed = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                             "core_schedules": modified}
                    try_add(groups, f"critical_path_reorder_{task}_{other}",
                            "reorder", fixed)
            if len(groups[task]) > 1:
                split = self._split_task(groups, task)
                if split:
                    pieces, _ = split
                    try_add(pieces, f"critical_path_split_{task}", "split")
        if problem == 1:
            ranked = [self._rank(row) for row in proposals.values()]
            ranked.sort(key=lambda row: (row["risk_objective"], row["hash"]))
        else:
            from common.scene_cost import score_plan
            ranked = list(proposals.values())
            for row in ranked:
                row["scene_proxy"] = score_plan(
                    self.g, row["plan"], problem, settings)
            ranked.sort(key=lambda row: (row["scene_proxy"]["objective"],
                                         row["hash"]))
        return ranked[:limit]

    def feedback_neighbors(self, chosen, official_result, *, limit=10):
        """Propose legal Task changes using the official longest-core timeline."""
        feedback_started = time.monotonic()
        groups, plan = chosen["groups"], chosen["plan"]
        proposals = {}
        def add(candidate):
            if candidate:
                proposals.setdefault(candidate["hash"], candidate)
        # Split the official critical tail and the largest Step1 memory risk.
        timelines = official_result.get("per_core_timeline", [])
        if timelines:
            worst_core = max(timelines, key=lambda core: max(
                (t.get("end",0) for t in core.get("tasks", [])), default=0))
            timed = worst_core.get("tasks", [])
            critical = max(timed, key=lambda t: t.get("end",0)) if timed else None
            heavy = max(timed, key=lambda t: t.get("duration",0)) if timed else None
        else:
            worst_core = None; critical = heavy = None
        split_targets = []
        for task in (critical, heavy):
            if task:
                split_targets.append(task.get("subgraph_id", task.get("task_id")))
        split_targets += [row["task"] for row in chosen.get("risky_tasks", [])[:2]]
        for task_id in dict.fromkeys(split_targets):
            if task_id is None or task_id >= len(groups):
                continue
            split = self._split_task(groups, task_id)
            if split:
                proposal, kind = split
                try:
                    add(self._candidate(proposal, chosen["label"] + f":{kind}:{task_id}",
                        {"operation": kind, "task": task_id,
                         "reason": "official critical tail or Step1 overflow"}))
                except Exception:
                    pass
        # Merge a few dependency-linked Tasks with the largest intermediate
        # Tensor boundary. Quotient DAG validation rejects non-convex merges.
        for a,b in self._merge_targets(groups):
            proposal = [list(g) for i,g in enumerate(groups) if i not in (a,b)]
            proposal.append(list(groups[a]) + list(groups[b]))
            try:
                add(self._candidate(proposal, chosen["label"] + f":merge:{a}:{b}",
                    {"operation": "merge", "tasks": [a,b],
                     "reason": "large intermediate Tensor boundary"}))
            except Exception:
                pass
        # Move the official longest-core tail/heavy Task, then try swapping
        # the final adjacent independent pair. Fixed-schedule validation checks
        # combined Task dependencies and core order for cycles.
        if worst_core is not None:
            source = worst_core["core_id"]
            for item in (critical, heavy):
                if item is None:
                    continue
                task = item.get("subgraph_id", item.get("task_id"))
                if task is None:
                    continue
                for target in range(self.k):
                    if target == source:
                        continue
                    orders = [list(x) for x in plan["core_schedules"]]
                    if task not in orders[source]:
                        continue
                    orders[source].remove(task); orders[target].append(task)
                    modified = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                                "core_schedules": orders}
                    add(self._candidate(groups, chosen["label"] + f":move:{task}:{source}->{target}",
                        {"operation": "move", "task": task, "from_core": source,
                         "to_core": target, "reason": "official longest core"}, modified))
            order = plan["core_schedules"][source]
            if len(order) >= 2:
                orders = [list(x) for x in plan["core_schedules"]]
                orders[source][-2], orders[source][-1] = orders[source][-1], orders[source][-2]
                modified = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                            "core_schedules": orders}
                add(self._candidate(groups, chosen["label"] + f":reorder:{source}",
                    {"operation": "reorder", "core": source,
                     "reason": "official longest core tail"}, modified))
        proxy_started = time.monotonic()
        ranked = [self._rank(row) for row in proposals.values()]
        ranked.sort(key=lambda r: (r["risk_objective"], r["task_count"], r["hash"]))
        self.last_feedback_phase_seconds = {
            "neighbor_generation": proxy_started - feedback_started,
            "proxy_ranking": time.monotonic() - proxy_started}
        return ranked[:limit]

    def feedback_neighbors_scene_b(self, chosen, official_result, *, problem,
                                   settings, limit=10):
        """Use merged-core Task feedback for Problems 2/3.

        The official timeline has one Task per core, so its task_id must not
        be interpreted as a subgraph id. Existing split/merge constructors
        are reused; moves and reorders preserve the current partition.
        """
        from common.scene_cost import score_plan
        feedback_started = time.monotonic()
        proxy_seconds = 0.0

        if problem not in (2, 3):
            raise ValueError(problem)
        groups, plan = chosen["groups"], chosen["plan"]
        orders = plan["core_schedules"]
        if len(orders) < 2:
            return []
        timeline = official_result.get("per_core_timeline", [])
        finish = [max((task.get("end", 0) for task in core.get("tasks", [])),
                      default=0) for core in timeline]
        work = [sum(self.g.ops[op].get("cycles", 0) for op in group)
                for group in groups]
        if len(finish) != self.k:
            finish = [sum(work[t] for t in row) for row in orders]
        heavy = max(range(self.k), key=lambda c: (finish[c], -c))
        light = min(range(self.k), key=lambda c: (finish[c], c))
        movement = official_result.get("data_movement_bytes", {})
        spill = movement.get("spill_added_copy_bytes", 0)
        copy = movement.get("partition_added_copy_bytes", 0)
        bottleneck = ("spill" if spill > copy and spill > 0 else
                      "communication" if copy > 0 and copy >= spill else
                      "core_load")
        proposals = {}

        def add(candidate, operation):
            nonlocal proxy_seconds
            if candidate is None or candidate["hash"] in proposals:
                return
            try:
                proxy_started = time.monotonic()
                scene = score_plan(self.g, candidate["plan"], problem, settings)
                proxy_seconds += time.monotonic() - proxy_started
            except (KeyError, ValueError):
                return
            candidate["scene_proxy"] = scene
            candidate["feedback_bottleneck"] = bottleneck
            candidate["feedback_operation"] = operation
            proposals[candidate["hash"]] = candidate

        # Existing partition edit: split a large subgraph on the busy core.
        # This can change Step1 order and local memory lifetimes.
        for task in sorted(orders[heavy], key=lambda t: (-work[t], t))[:2]:
            split = self._split_task(groups, task)
            if split:
                pieces, kind = split
                try:
                    add(self._candidate(pieces, f"scene_b_{kind}_{task}"), "split")
                except (ValueError, KeyError):
                    pass
            if bottleneck == "spill":
                for fraction in (0.25, 0.75):
                    variant = self._split_task_fraction_fixed(
                        groups, plan, task, fraction)
                    if variant:
                        pieces, fixed_plan = variant
                        try:
                            add(self._candidate(
                                pieces, f"scene_b_residency_split_{task}_{fraction}",
                                plan=fixed_plan), "split")
                        except (ValueError, KeyError):
                            pass

        if bottleneck == "spill":
            # Existing natural modules are the structural starting point.
            # Multiway topological cuts inside them expose different Step1
            # orders to the merged-core allocator. Probe several bounded
            # scales because Spill is not monotone in the number of cuts.
            position = self.base.op_pos
            for factor in (2, 4, 8):
                refined = []
                for module in self.base.modules:
                    ordered = sorted(module, key=position.__getitem__)
                    width = max(1, math.ceil(len(ordered) / factor))
                    refined.extend(ordered[i:i + width]
                                   for i in range(0, len(ordered), width))
                if len(refined) > 512:
                    continue
                try:
                    candidate = self._candidate(
                        refined, f"scene_b_spill_module_refine_{factor}")
                    if candidate:
                        candidate["feedback_scale"] = factor
                    add(candidate, "refine")
                except (ValueError, KeyError):
                    pass

        if heavy != light:
            for task in sorted(orders[heavy], key=lambda t: (-work[t], t))[:3]:
                for position in sorted({0, len(orders[light]) // 2,
                                        len(orders[light])}):
                    modified = [list(row) for row in orders]
                    modified[heavy].remove(task)
                    modified[light].insert(position, task)
                    fixed_plan = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                                  "core_schedules": modified}
                    add(self._candidate(groups, f"scene_b_move_{task}_{position}",
                                        plan=fixed_plan), "move")

        # The same-core order affects residency even without a COPY boundary.
        for core in sorted({heavy, light}):
            row = orders[core]
            for at in range(min(len(row) - 1, 3)):
                modified = [list(x) for x in orders]
                modified[core][at], modified[core][at + 1] = (
                    modified[core][at + 1], modified[core][at])
                fixed_plan = {"node_to_subgraph": dict(plan["node_to_subgraph"]),
                              "core_schedules": modified}
                add(self._candidate(groups, f"scene_b_reorder_{core}_{at}",
                                    plan=fixed_plan), "reorder")

        # Merge across a costly boundary only when it can change assignment
        # or local operation order; no same-core COPY saving is assumed.
        core_of = {task: core for core, row in enumerate(orders) for task in row}
        for a, b in self._merge_targets(groups):
            if core_of[a] == core_of[b]:
                continue
            merged = [list(g) for i, g in enumerate(groups) if i not in (a, b)]
            merged.append(list(groups[a]) + list(groups[b]))
            try:
                add(self._candidate(merged, f"scene_b_merge_{a}_{b}"), "merge")
            except (ValueError, KeyError):
                pass
            break

        preference = ({"refine": 0, "split": 1, "reorder": 2, "move": 3, "merge": 4}
                      if bottleneck == "spill" else
                      {"merge": 0, "move": 1, "split": 2, "reorder": 3, "refine": 4}
                      if bottleneck == "communication" else
                      {"move": 0, "reorder": 1, "split": 2, "merge": 3, "refine": 4})
        ranked = sorted(proposals.values(), key=lambda row: (
            preference[row["feedback_operation"]],
            row.get("feedback_scale", 0),
            row["scene_proxy"]["objective"], row["hash"]))
        self.last_feedback_phase_seconds = {
            "neighbor_generation": time.monotonic() - feedback_started - proxy_seconds,
            "proxy_ranking": proxy_seconds}
        return ranked[:limit]
