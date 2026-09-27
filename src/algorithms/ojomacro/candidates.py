"""OJOmacro constructor and official-feedback candidates."""
import time
from collections import defaultdict
from algorithms.ojomacro.search import OJOLNS
from common.evaluation import plan_digest


def construct_rows(graph, problem, cores, settings, common, *, limit,
                   search_seconds, diagnostics=None,
                   ojo_communication_boundary=False):
    rows = []
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
    return rows


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



def feedback_source(graph, groups, plan, problem, cores, settings,
                    common, *, limit, search_seconds, diagnostics=None,
                    ojo_communication_boundary=False):
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
    return source, stage
