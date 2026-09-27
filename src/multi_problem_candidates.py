"""Shared route constructors with separate scene-B/C selection and feedback."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

from aggregate_v2 import FastV1ModuleAggregator
from aggregate_v2plus import V2PlusAggregator
from evaluation_adapter import plan_digest
from ojo_lns import OJOLNS
from ramp_plus import RAMPPlus
from scene_cost import score_plan
from schedule import make_plan


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


def _ojo_scene_a_portfolio(rows, limit):
    """Spend scarce official slots on distinct constructor structures first.

    The caller may receive fewer than ``limit`` candidates. Its unused slots
    remain available for official-incumbent raw-Op feedback.
    """
    unique, hashes = [], set()
    for row in rows:
        digest = plan_digest(row["plan"])
        if digest not in hashes:
            unique.append(row)
            hashes.add(digest)
    if not unique or limit <= 0:
        return []
    macros = [row for row in unique if row["label"].startswith("macro_")]
    selected, chosen_hashes, scales = [], set(), set()

    def add(row):
        digest = plan_digest(row["plan"])
        if digest in chosen_hashes or len(selected) >= limit:
            return
        selected.append(row)
        chosen_hashes.add(digest)
        scales.add(len(row["groups"]).bit_length())

    if macros:
        add(min(macros, key=lambda row: (row["route_proxy"], row["label"])))
    constructors = [row for row in unique
                    if not row["label"].startswith(("repair_", "schedule_", "local_cpsat_"))]
    for row in constructors:
        if len(row["groups"]).bit_length() not in scales:
            add(row)
        if len(selected) >= min(3, limit):
            break
    if limit <= 3:
        return selected
    for row in unique:
        add(row)
    return selected


def route_candidates(graph, algorithm, problem, cores, settings, *,
                     limit, search_seconds, diagnostics=None,
                     ojo_communication_boundary=False):
    """Generate fresh candidates. Scene B/C use official ranking after proxy.

    OJO B/C constructors use the merged-core scene proxy. Official-guided
    raw-Op LNS is invoked separately after a verified incumbent is available.
    """
    bandwidth = settings["bandwidth"]
    scene_a = settings["scene_a"]
    common = {"bandwidth": bandwidth,
              "same_wait": scene_a["task_same_core_wait_cycles"],
              "cross_wait": scene_a["task_cross_core_wait_cycles"],
              "capacity": settings["capacity"]}
    rows = []
    started = time.monotonic()
    if algorithm == "V2plus":
        modules = FastV1ModuleAggregator(
            graph, cache_bytes=sum(settings["capacity"].values())).run()
        solved = V2PlusAggregator(graph, modules, cores, **common).run()[1]
        for item in solved["ranked_candidates"]:
            rows.append({"plan": item["plan"], "groups": item["groups"],
                         "label": item["source"], "source": "V2plus",
                         "search_stage": "constructor"})
    elif algorithm == "RAMPplus":
        phase_started = time.monotonic()
        route = RAMPPlus(graph, cores, **common)
        route_init_seconds = time.monotonic() - phase_started
        solved = route.initial_candidates(time_budget=max(1, search_seconds),
                                          limit=max(limit * 3, 8))
        if diagnostics is not None:
            diagnostics.update(route_init_seconds=route_init_seconds,
                               ramp_phase_seconds=solved["phase_seconds"],
                               generated_candidate_count=solved["all_ranked_count"],
                               natural_module_count=len(route.base.modules))
        for item in solved["candidates"]:
            rows.append({"plan": item["plan"], "groups": item["groups"],
                         "label": item["label"], "source": "RAMPplus",
                         "search_stage": "constructor"})
    elif algorithm == "OJOmacro":
        phase_started = time.monotonic()
        route = OJOLNS(graph, cores, **common, cpsat=False,
                       problem=problem, scene_settings=settings,
                       communication_boundary=ojo_communication_boundary)
        if problem == 1:
            solved = route.solve(iterations=60, seed_count=max(limit * 2, 8),
                                 time_budget=max(1, search_seconds),
                                 schedule_search=False, multiscale=True, macro=True)
            source = solved["candidates"]
            if diagnostics is not None:
                diagnostics["ojo_phase_seconds"] = solved["phase_seconds"]
                diagnostics["candidate_cache_hits"] = solved["candidate_cache_hits"]
        else:
            source = route.construct(macro=True)
            # Bound scene-specific scoring on large graphs. Include both old
            # raw-Op construction and every macro mechanism.
            by_family = defaultdict(list)
            for candidate in source:
                label = candidate["label"]
                family = label.split("_")[1] if label.startswith("macro_") else "legacy"
                by_family[family].append(candidate)
            source = []
            for family in sorted(by_family):
                source.extend(sorted(by_family[family],
                                     key=lambda r: (r["proxy_makespan"], r["hash"]))[:4])
        if diagnostics is not None:
            diagnostics.update(constructor_seconds=time.monotonic() - phase_started,
                               generated_candidate_count=len(source),
                               raw_op_count=len(graph.compute_ids),
                               candidate_cache_hits=route._candidate_cache_hits)
        for item in source:
            rows.append({"plan": item["plan"], "groups": item["groups"],
                         "label": item["label"], "source": "OJOmacro",
                         "route_proxy": item["proxy_makespan"],
                         "destroy_info": item.get("destroy_info"),
                         "search_stage": "scene_a_lns" if problem == 1 else "stage1_constructor"})
    else:
        raise ValueError(algorithm)
    if problem == 1:
        # Preserve each existing route's Scene-A ordering. The new cost proxy
        # is only used for the merged-core and L2 scenes. OJO receives one
        # structural macro representative before its LNS ranked candidates;
        # otherwise a small official quota can miss every macro scale.
        if algorithm == "OJOmacro":
            chosen = _ojo_scene_a_portfolio(rows, limit)
            if diagnostics is not None:
                diagnostics.update(total_seconds=time.monotonic() - started,
                                   selected_candidate_count=len(chosen),
                                   selected_labels=[row["label"] for row in chosen])
            return chosen
        chosen, seen = [], set()
        for row in rows:
            digest = json.dumps(row["plan"], sort_keys=True, separators=(",", ":"))
            if digest not in seen:
                chosen.append(row)
                seen.add(digest)
            if len(chosen) >= limit:
                break
        if diagnostics is not None:
            diagnostics.update(total_seconds=time.monotonic() - started,
                               selected_candidate_count=len(chosen))
        return chosen
    ranking_started = time.monotonic()
    chosen = _rank(graph, rows, problem, settings, limit)
    if diagnostics is not None:
        diagnostics.update(total_seconds=time.monotonic() - started,
                           scene_ranking_seconds=time.monotonic() - ranking_started,
                           selected_candidate_count=len(chosen))
    return chosen


def feedback_candidates(graph, plan, official_result, algorithm, problem,
                        cores, settings, *, limit, search_seconds,
                        exclude_plan_hashes=(), diagnostics=None,
                        ramp_critical_path=False,
                        ojo_communication_boundary=False):
    """Improve one officially verified Plan with the route's own neighborhood."""
    if algorithm == "V2plus" or limit <= 0:
        return []
    started = time.monotonic()
    members = defaultdict(list)
    for op, task in plan["node_to_subgraph"].items():
        members[int(task)].append(int(op))
    groups = [sorted(members[t]) for t in sorted(members)]
    common = {"bandwidth": settings["bandwidth"],
              "same_wait": settings["scene_a"]["task_same_core_wait_cycles"],
              "cross_wait": settings["scene_a"]["task_cross_core_wait_cycles"],
              "capacity": settings["capacity"]}
    if algorithm == "RAMPplus":
        route = RAMPPlus(graph, cores, **common)
        if diagnostics is not None:
            diagnostics["ramp_natural_module_seconds"] = route.base.phase_seconds.get(
                "natural_module_aggregation")
        incumbent = {"groups": groups, "plan": plan, "label": "official_incumbent"}
        if problem == 1:
            source = route.feedback_neighbors(incumbent, official_result,
                                              limit=max(limit * 3, 8))
        else:
            source = route.feedback_neighbors_scene_b(
                incumbent, official_result, problem=problem, settings=settings,
                limit=max(limit * 3, 8))
        if ramp_critical_path:
            critical_started = time.monotonic()
            targeted = route.critical_path_neighbors(
                incumbent, official_result, problem=problem, settings=settings,
                limit=min(3, max(1, limit)))
            existing = {base["hash"] for base in source}
            targeted = [row for row in targeted if row["hash"] not in existing]
            # Reserve one scarce official slot for the new source, while
            # retaining the existing feedback proposals as fallback.
            source = targeted[:1] + source if targeted else source
            if diagnostics is not None:
                diagnostics["critical_path_generated"] = len(targeted)
                diagnostics["critical_path_generation_seconds"] = (
                    time.monotonic() - critical_started)
        if diagnostics is not None:
            diagnostics["ramp_feedback_phase_seconds"] = route.last_feedback_phase_seconds
        stage = "ramp_official_feedback"
    elif algorithm == "OJOmacro":
        route = OJOLNS(graph, cores, **common, cpsat=False,
                       problem=problem, scene_settings=settings,
                       communication_boundary=ojo_communication_boundary)
        solved = route.solve(iterations=16, seed_count=max(limit * 4, 12),
                             time_budget=max(1, search_seconds),
                             incumbent={"groups": groups, "plan": plan},
                             schedule_search=(problem == 1), multiscale=True,
                             macro=True, feedback_only=True)
        source = solved["candidates"]
        if diagnostics is not None:
            diagnostics.update(ojo_phase_seconds=solved["phase_seconds"],
                               valid_repairs=solved["valid_repairs"],
                               destroy_attempts=solved["attempts"],
                               unique_pool_candidates=solved["candidate_count"],
                               candidate_cache_hits=solved["candidate_cache_hits"])
            diagnostics["communication_boundary_generated"] = sum(
                item["label"].endswith("communication_boundary")
                for item in source)
        stage = "op_lns_official_feedback"
    else:
        raise ValueError(algorithm)
    if diagnostics is not None:
        diagnostics.update(neighborhood_generation_seconds=time.monotonic() - started,
                           generated_candidate_count=len(source))
    original = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    excluded = set(exclude_plan_hashes)
    rows = []
    for item in source:
        candidate = item["plan"]
        if json.dumps(candidate, sort_keys=True, separators=(",", ":")) == original:
            continue
        if plan_digest(candidate) in excluded:
            continue
        rows.append({"plan": candidate, "groups": item["groups"],
                     "label": item["label"], "source": algorithm,
                     "route_proxy": item.get("proxy_makespan"),
                     "search_stage": stage,
                     "destroy_info": item.get("destroy_info"),
                     "feedback_bottleneck": item.get("feedback_bottleneck"),
                     "feedback_operation": item.get("feedback_operation"),
                     "block_info": item.get("block_info")})
    boundary = []
    if algorithm == "OJOmacro" and ojo_communication_boundary:
        boundary = sorted(
            (row for row in rows
             if row["label"].endswith("communication_boundary")),
            key=lambda row: (row["route_proxy"], row["label"]))
        if boundary:
            rows = [boundary[0]] + rows
    if problem == 1 or algorithm == "RAMPplus":
        distinct, seen = [], set()
        for row in rows:
            digest = plan_digest(row["plan"])
            if digest not in seen:
                distinct.append(row)
                seen.add(digest)
            if len(distinct) >= limit:
                break
        return distinct
    for row in rows:
        row["scene_proxy"] = score_plan(graph, row["plan"], problem, settings)
    # With a tiny official-evaluation quota, cover distinct raw-Op destroy
    # neighborhoods before trusting a close proxy ranking. Larger quotas also
    # admit a medium repair; constructor families were evaluated earlier.
    selected, seen, kinds = [], set(), set()
    if boundary:
        selected.append(boundary[0])
        seen.add(plan_digest(boundary[0]["plan"]))
        kinds.add("communication_boundary")
    for scale in ("small", "medium"):
        if scale == "medium" and limit < 3:
            continue
        repairs = sorted((row for row in rows
                          if row["label"].startswith(f"repair_{scale}_")),
                         key=lambda row: (row["scene_proxy"]["objective"],
                                          row["label"]))
        quota = min(2, limit) if scale == "small" else 1
        added = 0
        for row in repairs:
            kind = row["label"].split("_", 2)[-1]
            digest = plan_digest(row["plan"])
            if kind in kinds or digest in seen:
                continue
            selected.append(row)
            seen.add(digest)
            kinds.add(kind)
            added += 1
            if added >= quota or len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    for row in _rank(graph, rows, problem, settings, len(rows)):
        digest = plan_digest(row["plan"])
        if digest not in seen:
            selected.append(row)
            seen.add(digest)
        if len(selected) >= limit:
            break
    return selected


def ramp_block_candidates(graph, plan, official_result, problem, cores,
                          settings, *, limit, search_seconds,
                          exclude_plan_hashes=(), diagnostics=None):
    """Intensify the latest official RAMP+ incumbent after ordinary feedback."""
    if limit <= 0:
        return []
    started = time.monotonic()
    members = defaultdict(list)
    for op, task in plan["node_to_subgraph"].items():
        members[int(task)].append(int(op))
    groups = [sorted(members[t]) for t in sorted(members)]
    route = RAMPPlus(
        graph, cores, bandwidth=settings["bandwidth"],
        same_wait=settings["scene_a"]["task_same_core_wait_cycles"],
        cross_wait=settings["scene_a"]["task_cross_core_wait_cycles"],
        capacity=settings["capacity"])
    source, block_stats = route.block_reopt_neighbors(
        {"groups": groups, "plan": plan, "label": "official_incumbent"},
        official_result, problem=problem, settings=settings,
        limit=8, time_budget=min(10.0, max(1.0, search_seconds * 0.25)))
    excluded = set(exclude_plan_hashes)
    original = plan_digest(plan)
    rows, seen = [], set()
    for item in source:
        digest = plan_digest(item["plan"])
        if digest == original or digest in excluded or digest in seen:
            continue
        seen.add(digest)
        rows.append({"plan": item["plan"], "groups": item["groups"],
                     "label": item["label"], "source": "RAMPplus",
                     "search_stage": "ramp_block_reopt",
                     "feedback_bottleneck": item.get("feedback_bottleneck"),
                     "feedback_operation": "block_reopt",
                     "block_info": item["block_info"]})
    if diagnostics is not None:
        diagnostics.update(block_reopt=block_stats,
                           block_candidates_after_dedup=len(rows),
                           block_candidates_selected=min(limit, len(rows)),
                           generated_candidate_count=len(source),
                           neighborhood_generation_seconds=time.monotonic() - started)
    return rows[:limit]


def scene_b_feedback(graph, raw, plan, official_result, problem, settings,
                     *, limit=5):
    """Core-aware move/reorder proposals for the merged Task in P2/P3.

    Uses each core's single Task timeline and subgraph work, never a sequence
    of independent per-subgraph Task ids. Final legality is checked by the
    official plan parser and then by the selected official evaluator.
    """
    if problem not in (2, 3):
        return []
    orders = plan["core_schedules"]
    if len(orders) < 2:
        return []
    owner = {int(op): int(task) for op, task in plan["node_to_subgraph"].items()}
    work = defaultdict(int)
    for op in graph.compute_ids:
        work[owner[op]] += int(graph.ops[op].get("cycles", 0))
    timelines = official_result.get("per_core_timeline", [])
    finish = [max((task.get("end", 0) for task in core.get("tasks", [])), default=0)
              for core in timelines]
    if len(finish) != len(orders):
        finish = [sum(work[t] for t in row) for row in orders]
    heavy = max(range(len(orders)), key=lambda c: (finish[c], -c))
    light = min(range(len(orders)), key=lambda c: (finish[c], c))
    candidates = []
    def add(new_orders, label):
        new_plan = {"node_to_subgraph": plan["node_to_subgraph"],
                    "core_schedules": new_orders}
        try:
            from stub_multicore_cut_and_schedule import derive_multicore_plan
            derive_multicore_plan(raw, new_plan)
            proxy = score_plan(graph, new_plan, problem, settings)
            candidates.append({"plan": new_plan, "label": label,
                               "source": "scene_b_feedback", "search_stage": "feedback",
                               "scene_proxy": proxy})
        except (ValueError, KeyError, RuntimeError):
            pass
    if heavy != light:
        for task in sorted(orders[heavy], key=lambda t: -work[t])[:3]:
            for position in (0, len(orders[light]) // 2, len(orders[light])):
                new_orders = [list(row) for row in orders]
                new_orders[heavy].remove(task)
                new_orders[light].insert(position, task)
                add(new_orders, f"move_subgraph_{task}_{heavy}_to_{light}_{position}")
    # The order of subgraphs within one merged Task changes Step1/2 residency.
    for core in sorted({heavy, light}):
        row = orders[core]
        for position in (0, max(0, len(row) // 2 - 1), max(0, len(row) - 2)):
            if position + 1 < len(row):
                new_orders = [list(x) for x in orders]
                new_orders[core][position:position + 2] = reversed(
                    new_orders[core][position:position + 2])
                add(new_orders, f"reorder_core_{core}_{position}")
    candidates.sort(key=lambda row: (row["scene_proxy"]["objective"], row["label"]))
    return candidates[:limit]
