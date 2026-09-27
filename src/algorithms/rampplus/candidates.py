"""RAMPplus constructor and official-feedback candidates."""
import time
from collections import defaultdict
from algorithms.rampplus.algorithm import RAMPPlus
from common.evaluation import plan_digest


def construct_rows(graph, cores, common, *, limit, search_seconds, diagnostics=None):
    rows = []
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
    return rows


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



def feedback_source(graph, groups, plan, official_result, problem, cores,
                    settings, common, *, limit, diagnostics=None,
                    ramp_critical_path=False):
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
    return source, stage
