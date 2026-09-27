"""Dispatch the three public algorithms and their official-feedback candidates."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

from algorithms.v2plus.algorithm import construct_rows as construct_v2plus_rows
from common.evaluation import plan_digest
from algorithms.ojomacro.candidates import (construct_rows as construct_ojo_rows,
                                             feedback_source as ojo_feedback_source,
                                             _ojo_scene_a_portfolio)
from algorithms.rampplus.algorithm import (construct_rows as construct_rampplus_rows,
                                            feedback_source as ramp_feedback_source)
from common.scene_cost import score_plan
from common.schedule import make_plan


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
        rows = construct_v2plus_rows(graph, cores, settings, common)
    elif algorithm == "RAMPplus":
        rows = construct_rampplus_rows(
            graph, cores, common, limit=limit, search_seconds=search_seconds,
            diagnostics=diagnostics)
    elif algorithm == "OJOmacro":
        rows = construct_ojo_rows(
            graph, problem, cores, settings, common, limit=limit,
            search_seconds=search_seconds, diagnostics=diagnostics,
            ojo_communication_boundary=ojo_communication_boundary)
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
        source, stage = ramp_feedback_source(
            graph, groups, plan, official_result, problem, cores,
            settings, common, limit=limit, diagnostics=diagnostics,
            ramp_critical_path=ramp_critical_path)
    elif algorithm == "OJOmacro":
        source, stage = ojo_feedback_source(
            graph, groups, plan, problem, cores, settings, common,
            limit=limit, search_seconds=search_seconds,
            diagnostics=diagnostics,
            ojo_communication_boundary=ojo_communication_boundary)
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
